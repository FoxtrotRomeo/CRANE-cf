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

import pickle
import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from job_cf_factory import LABEL_CLASSES, TS_NAME, build_sepsis_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import MultimodalConsensusRetrieval, EarlyFusionNN, ModalityWisePrototypeSynthesis
from cf_lib.unimodal import TabularNN
from cf_lib.counterfactual_evaluation_helpers import compute_tau_c, fit_plausibility_normalizer, fit_proximity_normalizer
from cf_lib.counterfactual_helpers import find_k_closest_latent


class _SepsisIFModel(nn.Module):
    """Intermediate-fusion encoder — mirrors SepsisIFModel in train_intermediate_fusion.py."""

    def __init__(self, n_ts=53, n_st=47, ts_h=32, st_h=32, fused_h=32, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_ts, hidden_size=ts_h, batch_first=True)
        self.ts_drop = nn.Dropout(drop)
        self.static_enc = nn.Sequential(nn.Linear(n_st, st_h), nn.ReLU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(
            nn.Linear(ts_h + st_h, fused_h), nn.Tanh(),
            nn.BatchNorm1d(fused_h, momentum=0.01, eps=0.001), nn.Dropout(drop),
        )
        self.output_layer = nn.Linear(fused_h, 1)

    def encode(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        return self.fusion(torch.cat([self.ts_drop(h.squeeze(0)), self.static_enc(x_static)], dim=1))

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output_layer(self.encode(x_ts, x_static)))


class _TSOnlyModel(nn.Module):
    """GRU branch — mirrors TSOnlyModel in train_late_fusion.py."""

    def __init__(self, n_ts=53, gru_h=32, dense_h=32, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_ts, hidden_size=gru_h, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.dense = nn.Linear(gru_h, dense_h)
        self.bn = nn.BatchNorm1d(dense_h, momentum=0.01, eps=0.001)
        self.out = nn.Linear(dense_h, 1)

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        h = torch.tanh(self.dense(self.drop(h.squeeze(0))))
        return torch.sigmoid(self.out(self.bn(h)))


class _StaticOnlyModel(nn.Module):
    """MLP branch — mirrors StaticOnlyModel in train_late_fusion.py."""

    def __init__(self, n_st=47, hidden=32, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_st, hidden), nn.Tanh(),
            nn.BatchNorm1d(hidden, momentum=0.01, eps=0.001), nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x_static))


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
    default=50,
    help="Candidates per fold. With 5 folds, 25 k_per_fold matches the old total of 100.",
)
parser.add_argument(
    "--folds",
    type=str,
    default="0,1,2,3,4",
    help="Comma-separated fold ids to process (default: 0,1,2,3,4).",
)
parser.add_argument(
    "--k-search-combined",
    type=int,
    default=300,
    help="k_search pool size for ModalityWisePrototypeSynthesis and MultimodalConsensusRetrieval (default: 300).",
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
parser.add_argument(
    "--fusion-strategy",
    type=str,
    choices=["early", "intermediate", "late"],
    default="early",
    help="Which fusion model to use for the predict_fn (default: early).",
)
parser.add_argument(
    "--no-fold-suffix",
    action="store_true",
    help=(
        "Do not append '_fold_N' to the run name when writing output directories. "
        "Use this to re-run into the same folder structure as an existing run."
    ),
)
parser.add_argument("--save-full", action="store_true", default=True)
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
parser.add_argument(
    "--eval-pkls",
    type=str,
    default=None,
    metavar="DIR[,DIR,...]",
    help=(
        "Instead of generating new counterfactuals, re-evaluate saved "
        "combo_*_results.pkl files at multiple k values. "
        "Comma-separated list of run directories (fold sub-dirs, each containing "
        "pkls + summary.jsonl). Results written to <first-dir>/k_ablation_summary.json."
    ),
)
parser.add_argument(
    "--k-values",
    type=str,
    default="1,5,10,20,50",
    help="Comma-separated k values for --eval-pkls mode (default: 1,5,10,20,50).",
)
args = parser.parse_args()

_fusion_strategy = args.fusion_strategy
_default_run_names = {
    "early":        "fusion_early_deep_k50",
    "intermediate": "fusion_intermediate_rf_k50",
    "late":         "fusion_late_deep_k50",
}
_base_run_name = args.run_name or _default_run_names[_fusion_strategy]

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
    produced = build_sepsis_dataset(fold=fold, gpu=args.gpu, load_model=True,
                                     fusion_strategy=_fusion_strategy)
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

    if _fusion_strategy == "early":
        predict_fn = _build_predict_fn(dataset, model, device=device, ts_name=TS_NAME)
    elif _fusion_strategy == "intermediate":
        _data_dir = Path(__file__).resolve().parent / "data"
        _if_enc = _SepsisIFModel()
        _if_enc.load_state_dict(
            torch.load(_data_dir / f"best_model_intermediate_fusion_mlp_fold{fold}.pt",
                       map_location="cpu")
        )
        _if_enc.to(device).eval()
        with open(_data_dir / f"best_model_intermediate_fusion_rf_fold{fold}.pkl", "rb") as _fh:
            _if_rf = pickle.load(_fh)

        def _batch_latents_if(X_static, X_ts, *, _enc=_if_enc, _dev=device, _bs=128):
            rows = []
            for _s in range(0, len(X_static), _bs):
                _e = min(_s + _bs, len(X_static))
                with torch.no_grad():
                    _lat = _enc.encode(
                        torch.tensor(X_ts[_s:_e], dtype=torch.float32, device=_dev),
                        torch.tensor(X_static[_s:_e], dtype=torch.float32, device=_dev),
                    ).cpu().numpy()
                rows.append(_lat)
            return np.vstack(rows).astype(np.float32)

        _if_tr_lat = _batch_latents_if(dataset.X_train_static, dataset.X_train_ts[TS_NAME])
        _if_te_lat = _batch_latents_if(dataset.X_test_static, dataset.X_test_ts[TS_NAME])
        _if_cache: dict = {}
        _if_lock = threading.Lock()
        for _i, _pred in enumerate(_if_rf.predict(_if_tr_lat).astype(float)):
            _key = (dataset.X_train_static[_i].astype(np.float32).tobytes(),
                    dataset.X_train_ts[TS_NAME][_i].astype(np.float32).tobytes())
            _if_cache[_key] = float(_pred)

        def predict_fn(x_tab, x_ts, _text=None, *, _enc=_if_enc, _clf=_if_rf,
                       _lock=_if_lock, _cache=_if_cache, _dev=device):
            x_tab_arr = np.asarray(x_tab, dtype=np.float32)
            x_ts_arr = _unwrap_ts_candidate(x_ts, TS_NAME)
            _key = (x_tab_arr.tobytes(), x_ts_arr.tobytes())
            _hit = _cache.get(_key)
            if _hit is not None:
                return _hit
            with _lock:
                with torch.no_grad():
                    _lat = _enc.encode(
                        torch.tensor(x_ts_arr[None, :, :], dtype=torch.float32, device=_dev),
                        torch.tensor(x_tab_arr[None, :], dtype=torch.float32, device=_dev),
                    ).cpu().numpy()
            _label = float(_clf.predict(_lat)[0])
            _cache[_key] = _label
            return _label

    elif _fusion_strategy == "late":
        _data_dir = Path(__file__).resolve().parent / "data"
        _lf_mts = _TSOnlyModel()
        _lf_mts.load_state_dict(
            torch.load(_data_dir / f"best_model_late_fusion_mlp_ts_fold{fold}.pt",
                       map_location="cpu")
        )
        _lf_mts.to(device).eval()
        _lf_mst = _StaticOnlyModel()
        _lf_mst.load_state_dict(
            torch.load(_data_dir / f"best_model_late_fusion_mlp_static_fold{fold}.pt",
                       map_location="cpu")
        )
        _lf_mst.to(device).eval()
        _lf_cache: dict = {}
        _lf_lock = threading.Lock()
        _X_tr_st = np.asarray(dataset.X_train_static, dtype=np.float32)
        _X_tr_ts = np.asarray(dataset.X_train_ts[TS_NAME], dtype=np.float32)
        for _s in range(0, len(_X_tr_st), 128):
            _e = min(_s + 128, len(_X_tr_st))
            with torch.no_grad():
                _p_t = _lf_mts(torch.tensor(_X_tr_ts[_s:_e], dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
                _p_s = _lf_mst(torch.tensor(_X_tr_st[_s:_e], dtype=torch.float32, device=device)).squeeze(1).cpu().numpy()
            for _i, _prob in enumerate((_p_t + _p_s) / 2.0):
                _key = (_X_tr_st[_s + _i].tobytes(), _X_tr_ts[_s + _i].tobytes())
                _lf_cache[_key] = float(_prob >= 0.5)

        def predict_fn(x_tab, x_ts, _text=None, *, _mts=_lf_mts, _mst=_lf_mst,
                       _lock=_lf_lock, _cache=_lf_cache, _dev=device):
            x_tab_arr = np.asarray(x_tab, dtype=np.float32)
            x_ts_arr = _unwrap_ts_candidate(x_ts, TS_NAME)
            _key = (x_tab_arr.tobytes(), x_ts_arr.tobytes())
            _hit = _cache.get(_key)
            if _hit is not None:
                return _hit
            with _lock:
                with torch.no_grad():
                    _p_t = _mts(torch.tensor(x_ts_arr[None, :, :], dtype=torch.float32, device=_dev)).squeeze().cpu().item()
                    _p_s = _mst(torch.tensor(x_tab_arr[None, :], dtype=torch.float32, device=_dev)).squeeze().cpu().item()
            _label = float((_p_t + _p_s) / 2.0 >= 0.5)
            _cache[_key] = _label
            return _label

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
    prox_norm = fit_proximity_normalizer(
        X_tab_obs=dataset.X_train_static,
        X_ts_obs=dataset.X_train_ts[TS_NAME],
        y_train=dataset.y_train,
    )

    torch_latents = None
    if _fusion_strategy == "early":
        try:
            torch_latents = _compute_torch_latents(
                model,
                dataset,
                device=device,
                ts_name=TS_NAME,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  [warn] Could not compute fold-{fold} latents: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    elif _fusion_strategy == "intermediate":
        torch_latents = (_if_tr_lat, _if_te_lat)

    def _objectives_kwargs_factory(_text_cfg, _image_cfg, *, _primary_tab=dataset.primary_tabular_name):
        return {
            "y_target": target_value,
            "predict_fn": predict_fn,
            "proximity_normalizer": prox_norm,
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
            "MPS": ModalityWisePrototypeSynthesis(
                k=args.k_per_fold,
                k_search=args.k_search_combined,
                static_dist_fn=static_dist,
            ),
            "MC-R": MultimodalConsensusRetrieval(
                k=args.k_per_fold,
                k_search=args.k_search_combined,
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
    if args.no_fold_suffix:
        fold_run_name = _base_run_name
    else:
        fold_run_name = f"{_base_run_name}_fold_{fold}"

    if args.eval_pkls:
        from evaluate_k_ablation import evaluate_k_ablation
        # Auto-derive the per-fold run directory from the base eval dir.
        # Matches the path written during generation:
        #   {eval_pkls}/fold_{fold}/{fold_run_name}/
        _eval_base = Path(args.eval_pkls)
        _eval_run_dir = _eval_base / f"fold_{fold}" / fold_run_name
        if not _eval_run_dir.exists():
            print(f"  [k-eval] Skipping fold {fold}: {_eval_run_dir} not found.")
        else:
            _k_vals = [int(x) for x in args.k_values.split(",")]
            evaluate_k_ablation(
                run_dirs=[_eval_run_dir],
                dataset=dataset,
                objectives_kwargs_factory=_objectives_kwargs_factory,
                k_values=_k_vals,
                output_path=_eval_run_dir / f"k_ablation_summary_k{'_'.join(str(k) for k in _k_vals)}.json",
            )
    else:
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
