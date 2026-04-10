"""Counterfactual ablation for the sepsis folds.

Runs one ablation per fold so train/test splits stay fully separate and each
fold always uses its own trained classifier for outcome evaluation.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from job_cf_factory import LABEL_CLASSES, TS_NAME, build_sepsis_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import CombinedNN, EarlyFusionNN, FrankensteinNN
from cf_lib.unimodal import TabularNN
from counterfactual_evaluation_helpers import compute_tau_c, fit_plausibility_normalizer
from counterfactual_helpers import find_k_closest_latent


def _parse_int_csv(raw: str) -> List[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _metric_to_static_dist_fn(metric: str):
    if metric == "manhattan":
        return manhattan_distances
    return euclidean_distances


def _unwrap_ts_candidate(x_ts, ts_name: str) -> np.ndarray:
    if isinstance(x_ts, dict):
        if ts_name in x_ts:
            return np.asarray(x_ts[ts_name], dtype=np.float32)
        if len(x_ts) == 1:
            return np.asarray(next(iter(x_ts.values())), dtype=np.float32)
        raise ValueError(f"Expected TS modality '{ts_name}', got {sorted(x_ts)}.")
    return np.asarray(x_ts, dtype=np.float32)


class PyTorchLatentNN(CounterfactualGenerator):
    """Nearest-neighbour search in the sepsis model's latent space."""

    def __init__(
        self,
        *,
        k: int,
        train_latents: np.ndarray,
        test_latents: np.ndarray,
        distance_metric: str = "euclidean",
    ):
        self.k = k
        self.train_latents = train_latents
        self.test_latents = test_latents
        self.distance_metric = distance_metric

    def generate(self, dataset, sample_idx: int, model=None, target_value: int = 0, k: Optional[int] = None):
        k = self.k if k is None else k
        indices, _ = find_k_closest_latent(
            X_train_latent=self.train_latents,
            y_train=dataset.y_train,
            X_test_latent=self.test_latents,
            selected_test_indices=[sample_idx],
            target_value=target_value,
            k=k,
            distance_metric=self.distance_metric,
        )
        return TabularNN._materialize(
            indices,
            sample_idx,
            dataset,
            distance_metric_label=self.distance_metric,
        )


def _compute_torch_latents(
    model,
    dataset,
    *,
    device: str,
    ts_name: str,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture the penultimate representation before the output layer."""
    model.eval()
    buf: Dict[str, np.ndarray] = {}

    def _hook(_module, inputs, _output):
        buf["z"] = inputs[0].detach().cpu().numpy()

    handle = model.output_layer.register_forward_hook(_hook)

    def _run(X_static: np.ndarray, X_ts: np.ndarray) -> np.ndarray:
        rows = []
        for start in range(0, len(X_static), batch_size):
            end = min(start + batch_size, len(X_static))
            xb_static = torch.tensor(X_static[start:end], dtype=torch.float32, device=device)
            xb_ts = torch.tensor(X_ts[start:end], dtype=torch.float32, device=device)
            with torch.no_grad():
                model(xb_ts, xb_static)
            rows.append(buf["z"].copy())
        return np.vstack(rows).astype(np.float32)

    try:
        train_latents = _run(dataset.X_train_static, dataset.X_train_ts[ts_name])
        test_latents = _run(dataset.X_test_static, dataset.X_test_ts[ts_name])
    finally:
        handle.remove()

    return train_latents, test_latents


def _build_predict_fn(dataset, model, *, device: str, ts_name: str):
    """Return a cached fold-local predict_fn for outcome evaluation."""
    model.eval()
    cache: Dict[tuple[bytes, bytes], float] = {}
    lock = threading.Lock()

    X_train_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    X_train_ts = np.asarray(dataset.X_train_ts[ts_name], dtype=np.float32)

    pred_rows = []
    for start in range(0, len(X_train_static), 128):
        end = min(start + 128, len(X_train_static))
        xb_static = torch.tensor(X_train_static[start:end], dtype=torch.float32, device=device)
        xb_ts = torch.tensor(X_train_ts[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            y_proba = model(xb_ts, xb_static).squeeze(1).detach().cpu().numpy()
        pred_rows.append(y_proba.astype(np.float32))
    train_pred = np.concatenate(pred_rows, axis=0)
    for i, pred in enumerate(train_pred):
        key = (X_train_static[i].tobytes(), X_train_ts[i].tobytes())
        cache[key] = float(pred >= 0.5)

    def _predict_fn(x_tab, x_ts, _text_unused=None):
        x_tab_arr = np.asarray(x_tab, dtype=np.float32)
        x_ts_arr = _unwrap_ts_candidate(x_ts, ts_name)
        key = (x_tab_arr.tobytes(), x_ts_arr.tobytes())
        hit = cache.get(key)
        if hit is not None:
            return hit

        with lock:
            xb_static = torch.tensor(x_tab_arr[None, :], dtype=torch.float32, device=device)
            xb_ts = torch.tensor(x_ts_arr[None, :, :], dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = float(model(xb_ts, xb_static).squeeze().detach().cpu().item())
        label = float(pred >= 0.5)
        cache[key] = label
        return label

    return _predict_fn


parser = argparse.ArgumentParser(
    description="Death->No-death counterfactual ablation on the sepsis folds."
)
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument(
    "--k-per-fold",
    type=int,
    default=25,
    help="Candidates per fold. With 5 folds, 25 k_per_fold matches the old total of 100.",
)
parser.add_argument(
    "--folds",
    type=str,
    default="0,1,2,3,4",
    help="Comma-separated fold ids to process (default: 0,1,2,3,4).",
)
parser.add_argument(
    "--max-samples",
    type=int,
    default=None,
    help="Optional cap on how many source-class test samples to process per fold.",
)
parser.add_argument("--max-combinations", type=int, default=None)
parser.add_argument("--n-jobs", type=int, default=1)
parser.add_argument(
    "--output-dir",
    type=str,
    default=str(Path(__file__).parent / "data" / "ablation_runs"),
)
parser.add_argument(
    "--ts-threshold-mode",
    type=str,
    choices=["zero", "constant", "mad"],
    default="zero",
    help=(
        "How to build per-channel TS sparsity thresholds: "
        "'zero' = all zeros, 'constant' = same scalar for every channel, "
        "'mad' = channel-specific factor * MAD as in the old behaviour."
    ),
)
parser.add_argument(
    "--ts-change-threshold",
    type=float,
    default=0.0,
    help="Scalar TS threshold used when --ts-threshold-mode=constant (default: 0.0).",
)
parser.add_argument(
    "--ts-threshold-factor",
    type=float,
    default=1.1,
    help="Multiplier for per-channel MAD thresholds when --ts-threshold-mode=mad (default: 1.1).",
)
parser.add_argument("--run-name", type=str, default=None)
parser.add_argument("--save-full", action="store_true")
parser.add_argument(
    "--source-class",
    type=str,
    default="death",
    help=f"Class to explain (default: death). Choices: {LABEL_CLASSES}.",
)
parser.add_argument(
    "--target-class",
    type=str,
    default="no_death",
    help=f"Counterfactual target class (default: no_death). Choices: {LABEL_CLASSES}.",
)
args = parser.parse_args()

folds = _parse_int_csv(args.folds)
if not folds:
    raise ValueError("No folds provided.")

label_classes = list(LABEL_CLASSES)
lc_lower = [name.lower() for name in label_classes]
source_class = args.source_class.lower()
target_class = args.target_class.lower()
if source_class not in lc_lower:
    raise ValueError(f"'{source_class}' not in label classes: {label_classes}.")
if target_class not in lc_lower:
    raise ValueError(f"'{target_class}' not in label classes: {label_classes}.")
if source_class == target_class:
    raise ValueError("--source-class and --target-class must be different.")

source_value = lc_lower.index(source_class)
target_value = lc_lower.index(target_class)

tab_metrics = ["euclidean", "manhattan"]
ts_metrics = ["dtw", "euclidean", "lcss"]
dtw_windows = [0.10]
total_k = args.k_per_fold * len(folds)

print("Running fold-wise sepsis distance ablation ...")
print(f"  Folds        : {folds}")
print(f"  Source       : {source_value} ({label_classes[source_value]})")
print(f"  Target       : {target_value} ({label_classes[target_value]})")
print(f"  k_per_fold   : {args.k_per_fold}")
print(f"  Total k      : {total_k}")
print(f"  Tab metrics  : {tab_metrics}")
print(f"  TS metrics   : {ts_metrics}")
print(f"  TS mode      : {args.ts_threshold_mode}")
if args.ts_threshold_mode == "constant":
    print(f"  TS threshold : {args.ts_change_threshold}")
elif args.ts_threshold_mode == "mad":
    print(f"  TS factor    : {args.ts_threshold_factor}")
print()

total_samples_used = 0

for fold in folds:
    produced = build_sepsis_dataset(fold=fold, gpu=args.gpu, load_model=True)
    dataset = produced["dataset"]
    model = produced["model"]
    device = produced["torch_device"]
    y_pred = produced["y_pred"]

    if y_pred is None:
        raise FileNotFoundError(
            f"{produced['fold_dir'] / 'y_pred.npy'} not found - run sepsis/evaluate.py first."
        )

    sample_indices = [int(i) for i, pred in enumerate(y_pred) if int(pred) == source_value]
    if args.max_samples is not None:
        sample_indices = sample_indices[: args.max_samples]

    print(f"Fold {fold}: {len(sample_indices)} samples predicted as '{label_classes[source_value]}'")
    if not sample_indices:
        print(f"  Skipping fold {fold} because there are no source-class predictions.")
        continue
    total_samples_used += len(sample_indices)

    predict_fn = _build_predict_fn(dataset, model, device=device, ts_name=TS_NAME)
    if args.ts_threshold_mode == "zero":
        tau_c = np.zeros(dataset.X_train_ts[TS_NAME].shape[-1], dtype=np.float32)
    elif args.ts_threshold_mode == "constant":
        tau_c = np.full(
            dataset.X_train_ts[TS_NAME].shape[-1],
            float(args.ts_change_threshold),
            dtype=np.float32,
        )
    else:
        tau_c = compute_tau_c(
            dataset.X_train_ts[TS_NAME],
            factor=float(args.ts_threshold_factor),
        ).astype(np.float32)
    ts_segments = [(0, int(dataset.X_train_ts[TS_NAME].shape[1]))]
    plaus_norm = fit_plausibility_normalizer(
        X_tab_obs=dataset.X_train_static,
        X_ts_obs=dataset.X_train_ts[TS_NAME],
        y_train=dataset.y_train,
        target_value=target_value,
    )

    torch_latents = None
    try:
        torch_latents = _compute_torch_latents(
            model,
            dataset,
            device=device,
            ts_name=TS_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] Could not compute fold-{fold} latents: {type(exc).__name__}: {exc}")

    def _objectives_kwargs_factory(_text_cfg, _image_cfg, *, _primary_tab=dataset.primary_tabular_name):
        return {
            "y_target": target_value,
            "predict_fn": predict_fn,
            "tabular_objective_contexts": {
                (_primary_tab or "__primary__"): {
                    "plausibility_normalizer": {
                        "lof": plaus_norm.get("tab_lof"),
                        "low": plaus_norm.get("tab_low", 0.0),
                        "high": plaus_norm.get("tab_high", 1.0),
                    }
                }
            },
            "ts_objective_contexts": {
                TS_NAME: {
                    "segments": ts_segments,
                    "tau_c": tau_c,
                    "rho": 0.0,
                    "plausibility_normalizer": {
                        "lof": plaus_norm.get("ts_lof"),
                        "low": plaus_norm.get("ts_low", 0.0),
                        "high": plaus_norm.get("ts_high", 1.0),
                    },
                }
            },
        }

    k_search = min(50, args.k_per_fold * 5)

    def _multimodal_generators_factory(
        tab_cfg,
        ts_cfg,
        text_cfg,
        text_backend_kwargs,
        image_cfg,
        image_backend_kwargs,
    ):
        del text_cfg, text_backend_kwargs, image_cfg, image_backend_kwargs
        primary_tab = dataset.primary_tabular_name
        tab_metric = (tab_cfg or {}).get(primary_tab, "euclidean")
        static_dist = _metric_to_static_dist_fn(tab_metric)

        extras = {
            "Frankenstein": FrankensteinNN(
                k=args.k_per_fold,
                k_search=k_search,
                static_dist_fn=static_dist,
            ),
            "Combined": CombinedNN(
                k=args.k_per_fold,
                k_search=k_search,
                static_dist_fn=static_dist,
            ),
            "EarlyFusion": EarlyFusionNN(
                k=args.k_per_fold,
                distance_metric=tab_metric,
            ),
        }
        if torch_latents is not None:
            extras["IntermediateFusion"] = PyTorchLatentNN(
                k=args.k_per_fold,
                train_latents=torch_latents[0],
                test_latents=torch_latents[1],
                distance_metric=tab_metric,
            )
        return extras

    fold_output_dir = str(Path(args.output_dir) / f"fold_{fold}")
    fold_run_name = None if args.run_name is None else f"{args.run_name}_fold_{fold}"

    run_distance_ablation(
        dataset=dataset,
        model=None,
        sample_indices=sample_indices,
        target_value=target_value,
        k=args.k_per_fold,
        tab_metrics=tab_metrics,
        ts_metrics=ts_metrics,
        dtw_windows=dtw_windows,
        text_encoders=[],
        image_encoders=[],
        output_dir=fold_output_dir,
        run_name=fold_run_name,
        save_full=args.save_full,
        max_combinations=args.max_combinations,
        n_jobs=args.n_jobs,
        objectives_kwargs_factory=_objectives_kwargs_factory,
        extra_generators_factory=_multimodal_generators_factory,
    )

print()
print(f"Total samples used for counterfactual generation: {total_samples_used}")
