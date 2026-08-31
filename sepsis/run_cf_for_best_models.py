"""
Check whether a distance ablation has been run for the best model in each
fusion strategy, and launch missing ablations (per fold).

Reads: sepsis/data/fusion_model_registry.json  (written by evaluate_fusion_models.py)
Writes: sepsis/data/ablation_runs/fold_{N}/<ablation_run_name>/summary.json

For each fusion strategy the script:
  1. Reads the registry to find the best model type (highest pooled macro-F1).
  2. Iterates over all N_FOLDS and checks whether
       data/ablation_runs/fold_{fold}/<ablation_run_name>/summary.json
     already exists.
  3. For missing folds, loads the fold-specific model, builds a predict_fn, and
     runs the distance ablation with the same metric set as run_cf_ablation.py.

Run
---
    python sepsis/run_cf_for_best_models.py --gpu 0
    python sepsis/run_cf_for_best_models.py --gpu 0 --dry-run   # check only
    python sepsis/run_cf_for_best_models.py --gpu 0 --fold 0    # single fold
    python sepsis/run_cf_for_best_models.py --gpu 0 --strategies intermediate,late
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make cf_lib and examples importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from job_cf_factory import LABEL_CLASSES, TS_NAME, build_sepsis_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import CombinedNN, EarlyFusionNN, FrankensteinNN
from cf_lib.counterfactual_evaluation_helpers import compute_tau_c, fit_plausibility_normalizer
from cf_lib.counterfactual_helpers import find_k_closest_latent

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run missing fold-level CF ablations for each fusion strategy."
)
parser.add_argument("--gpu",          type=int,  default=None)
parser.add_argument("--k",            type=int,  default=50)
parser.add_argument("--max-samples",  type=int,  default=None)
parser.add_argument("--n-jobs",       type=int,  default=1)
parser.add_argument("--n-folds",      type=int,  default=5)
parser.add_argument("--fold",         type=int,  default=None,
                    help="Process only this fold (default: all folds).")
parser.add_argument("--output-dir",   type=str,
                    default=str(Path(__file__).resolve().parent / "data" / "ablation_runs"))
parser.add_argument("--source-class", type=str,  default="death")
parser.add_argument("--target-class", type=str,  default="no_death")
parser.add_argument("--save-full",    action="store_true", default=True)
parser.add_argument("--dry-run",      action="store_true",
                    help="Print what would be run without executing.")
parser.add_argument("--strategies",   type=str,  default=None,
                    help="Comma-separated subset of strategies (default: all).")
args = parser.parse_args()

DATA_DIR = Path(__file__).resolve().parent / "data"
N_FOLDS  = args.n_folds
folds    = [args.fold] if args.fold is not None else list(range(N_FOLDS))

DEVICE = (
    f"cuda:{args.gpu}"
    if args.gpu is not None and torch.cuda.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

label_classes = list(LABEL_CLASSES)
lc_lower      = [c.lower() for c in label_classes]
source_value  = lc_lower.index(args.source_class.lower())
target_value  = lc_lower.index(args.target_class.lower())

# ---------------------------------------------------------------------------
# Load registry
# ---------------------------------------------------------------------------
REGISTRY_PATH = DATA_DIR / "fusion_model_registry.json"
if not REGISTRY_PATH.exists():
    raise FileNotFoundError(
        f"{REGISTRY_PATH} not found — run evaluate_fusion_models.py first."
    )
with open(REGISTRY_PATH) as fh:
    registry = json.load(fh)

# ---------------------------------------------------------------------------
# Resolve strategies
# ---------------------------------------------------------------------------
all_strategies = ["early", "intermediate", "late"]
if args.strategies:
    all_strategies = [s.strip() for s in args.strategies.split(",")]

# Build todo list: (strategy, best_key, entry, run_name, folds_needed)
todo: List[Tuple] = []
for strategy in all_strategies:
    best_key = registry["best_per_strategy"].get(strategy)
    if best_key is None:
        print(f"[{strategy}] No available model in registry — skipping.")
        continue
    entry    = registry["models"][best_key]
    run_name = entry["ablation_run_name"] + "_k50"

    missing_folds = []
    for fold in folds:
        summary = Path(args.output_dir) / f"fold_{fold}" / run_name / "summary.json"
        if not summary.exists():
            missing_folds.append(fold)

    if not missing_folds:
        print(f"[{strategy}] All fold ablations exist — skipping.")
        continue

    todo.append((strategy, best_key, entry, run_name, missing_folds))
    print(f"[{strategy}] run='{run_name}'  model_type={entry['model_type']}  "
          f"missing folds={missing_folds}")

if not todo:
    print("\nAll ablations are up to date.")
    sys.exit(0)

if args.dry_run:
    print("\nDry run — exiting without launching ablations.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Inline model definitions (must match training scripts exactly)
# ---------------------------------------------------------------------------

class SepsisIFModel(nn.Module):
    def __init__(self, n_ts_features=53, n_static_features=47,
                 ts_hidden=32, static_hidden=32, fused_hidden=32, dropout_rate=0.2):
        super().__init__()
        self.gru       = nn.GRU(input_size=n_ts_features, hidden_size=ts_hidden,
                                batch_first=True)
        self.ts_drop   = nn.Dropout(dropout_rate)
        self.static_enc = nn.Sequential(
            nn.Linear(n_static_features, static_hidden), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        self.fusion = nn.Sequential(
            nn.Linear(ts_hidden + static_hidden, fused_hidden), nn.Tanh(),
            nn.BatchNorm1d(fused_hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Linear(fused_hidden, 1)

    def encode(self, x_ts, x_static):
        _, h = self.gru(x_ts)
        return self.fusion(torch.cat([self.ts_drop(h.squeeze(0)),
                                       self.static_enc(x_static)], dim=1))

    def forward(self, x_ts, x_static):
        return torch.sigmoid(self.output_layer(self.encode(x_ts, x_static)))


class TSOnlyModel(nn.Module):
    def __init__(self, n_ts_features=53, gru_hidden=32, dense_hidden=32, dropout=0.2):
        super().__init__()
        self.gru   = nn.GRU(input_size=n_ts_features, hidden_size=gru_hidden, batch_first=True)
        self.drop  = nn.Dropout(dropout)
        self.dense = nn.Linear(gru_hidden, dense_hidden)
        self.bn    = nn.BatchNorm1d(dense_hidden, momentum=0.01, eps=0.001)
        self.out   = nn.Linear(dense_hidden, 1)

    def forward(self, x_ts):
        _, h = self.gru(x_ts)
        h = self.drop(h.squeeze(0))
        return torch.sigmoid(self.out(self.bn(torch.tanh(self.dense(h)))))


class StaticOnlyModel(nn.Module):
    def __init__(self, n_static_features=47, hidden=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static_features, hidden), nn.Tanh(),
            nn.BatchNorm1d(hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, x_static):
        return torch.sigmoid(self.net(x_static))


# ---------------------------------------------------------------------------
# IntermediateFusion NN (latent-space NN search)
# ---------------------------------------------------------------------------
class PyTorchLatentNN(CounterfactualGenerator):
    def __init__(self, *, k, train_latents, test_latents, distance_metric="euclidean"):
        self.k = k
        self.train_latents  = train_latents
        self.test_latents   = test_latents
        self.distance_metric = distance_metric

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None):
        from cf_lib.unimodal import TabularNN
        k = self.k if k is None else k
        indices, _ = find_k_closest_latent(
            X_train_latent=self.train_latents,
            y_train=dataset.y_train,
            X_test_latent=self.test_latents,
            selected_test_indices=[sample_idx],
            target_value=target_value, k=k,
            distance_metric=self.distance_metric,
        )
        return TabularNN._materialize(indices, sample_idx, dataset,
                                      distance_metric_label=self.distance_metric)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _metric_to_static_dist_fn(metric: str):
    return manhattan_distances if metric == "manhattan" else euclidean_distances


def _unwrap_ts(x_ts) -> np.ndarray:
    if isinstance(x_ts, dict):
        if TS_NAME in x_ts:
            return np.asarray(x_ts[TS_NAME], dtype=np.float32)
        return np.asarray(next(iter(x_ts.values())), dtype=np.float32)
    return np.asarray(x_ts, dtype=np.float32)


@torch.no_grad()
def _compute_latents(model, dataset, device, batch_size=128):
    """Extract penultimate representations from an IF model (before output_layer)."""
    model.eval()
    buf: Dict = {}

    def _hook(m, inp, out): buf["z"] = inp[0].detach().cpu().numpy()
    handle = model.output_layer.register_forward_hook(_hook)

    def _run(X_ts, X_static):
        rows = []
        for s in range(0, len(X_ts), batch_size):
            e = min(s + batch_size, len(X_ts))
            with torch.no_grad():
                model(
                    torch.tensor(X_ts[s:e],     dtype=torch.float32, device=device),
                    torch.tensor(X_static[s:e], dtype=torch.float32, device=device),
                )
            rows.append(buf["z"].copy())
        return np.vstack(rows).astype(np.float32)

    try:
        tr_lat = _run(dataset.X_train_ts[TS_NAME], dataset.X_train_static)
        te_lat = _run(dataset.X_test_ts[TS_NAME],  dataset.X_test_static)
    finally:
        handle.remove()
    return tr_lat, te_lat


# ===========================================================================
# Predict-fn builders
# ===========================================================================

def _build_predict_fn_early(dataset, model_path, device):
    """SepsisModel early-fusion predict_fn (identical to run_cf_ablation.py)."""
    from train_pytorch import SepsisModel
    model = SepsisModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    X_train_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    X_train_ts     = np.asarray(dataset.X_train_ts[TS_NAME], dtype=np.float32)
    cache: Dict[Tuple[bytes, bytes], float] = {}
    lock  = threading.Lock()

    for s in range(0, len(X_train_static), 128):
        e  = min(s + 128, len(X_train_static))
        xt = torch.tensor(X_train_ts[s:e],     dtype=torch.float32, device=device)
        xs = torch.tensor(X_train_static[s:e], dtype=torch.float32, device=device)
        with torch.no_grad():
            proba = model(xt, xs).squeeze(1).cpu().numpy()
        for i, p in enumerate(proba):
            key = (X_train_static[s+i].tobytes(), X_train_ts[s+i].tobytes())
            cache[key] = float(p >= 0.5)

    def _predict_fn(x_tab, x_ts, _text=None):
        xt = _unwrap_ts(x_ts)
        xs = np.asarray(x_tab, dtype=np.float32)
        hit = cache.get((xs.tobytes(), xt.tobytes()))
        if hit is not None:
            return hit
        with lock:
            with torch.no_grad():
                p = float(model(
                    torch.tensor(xt[None], dtype=torch.float32, device=device),
                    torch.tensor(xs[None], dtype=torch.float32, device=device),
                ).squeeze().item())
        return float(p >= 0.5)

    return _predict_fn, model


def _build_predict_fn_if_mlp(dataset, model_path, device):
    """SepsisIFModel predict_fn."""
    model = SepsisIFModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    X_train_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    X_train_ts     = np.asarray(dataset.X_train_ts[TS_NAME], dtype=np.float32)
    cache: Dict[Tuple[bytes, bytes], float] = {}
    lock  = threading.Lock()

    for s in range(0, len(X_train_static), 128):
        e  = min(s + 128, len(X_train_static))
        xt = torch.tensor(X_train_ts[s:e],     dtype=torch.float32, device=device)
        xs = torch.tensor(X_train_static[s:e], dtype=torch.float32, device=device)
        with torch.no_grad():
            proba = model(xt, xs).squeeze(1).cpu().numpy()
        for i, p in enumerate(proba):
            key = (X_train_static[s+i].tobytes(), X_train_ts[s+i].tobytes())
            cache[key] = float(p >= 0.5)

    def _predict_fn(x_tab, x_ts, _text=None):
        xt = _unwrap_ts(x_ts)
        xs = np.asarray(x_tab, dtype=np.float32)
        hit = cache.get((xs.tobytes(), xt.tobytes()))
        if hit is not None:
            return hit
        with lock:
            with torch.no_grad():
                p = float(model(
                    torch.tensor(xt[None], dtype=torch.float32, device=device),
                    torch.tensor(xs[None], dtype=torch.float32, device=device),
                ).squeeze().item())
        return float(p >= 0.5)

    return _predict_fn, model


def _build_predict_fn_sklearn_if(dataset, sk_path, mlp_path, device):
    """RF / GBT on IF latents — latent is extracted from the IF-MLP model."""
    if mlp_path is None or not Path(mlp_path).exists():
        raise FileNotFoundError(f"IF-MLP model file not found: {mlp_path}")
    if_model = SepsisIFModel()
    if_model.load_state_dict(torch.load(mlp_path, map_location=device))
    if_model.to(device).eval()

    with open(sk_path, "rb") as fh:
        sk_model = pickle.load(fh)

    X_train_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    X_train_ts     = np.asarray(dataset.X_train_ts[TS_NAME], dtype=np.float32)

    # Pre-extract training latents for the cache
    @torch.no_grad()
    def _extract(X_ts, X_static):
        rows = []
        for s in range(0, len(X_ts), 256):
            e  = min(s + 256, len(X_ts))
            xt = torch.tensor(X_ts[s:e],     dtype=torch.float32, device=device)
            xs = torch.tensor(X_static[s:e], dtype=torch.float32, device=device)
            rows.append(if_model.encode(xt, xs).cpu().numpy())
        return np.vstack(rows).astype(np.float32)

    Z_train = _extract(X_train_ts, X_train_static)
    lock    = threading.Lock()
    cache: Dict[Tuple[bytes, bytes], float] = {}
    for i in range(len(Z_train)):
        key = (X_train_static[i].tobytes(), X_train_ts[i].tobytes())
        cache[key] = float(sk_model.predict([Z_train[i]])[0])

    def _predict_fn(x_tab, x_ts, _text=None):
        xt  = _unwrap_ts(x_ts)
        xs  = np.asarray(x_tab, dtype=np.float32)
        hit = cache.get((xs.tobytes(), xt.tobytes()))
        if hit is not None:
            return hit
        with lock:
            with torch.no_grad():
                z = if_model.encode(
                    torch.tensor(xt[None], dtype=torch.float32, device=device),
                    torch.tensor(xs[None], dtype=torch.float32, device=device),
                ).cpu().numpy()
        return float(sk_model.predict(z)[0])

    return _predict_fn, if_model


def _build_predict_fn_late_deep(dataset, ts_path, static_path, device):
    """TSOnlyModel + StaticOnlyModel, averaged probability."""
    model_ts = TSOnlyModel()
    model_ts.load_state_dict(torch.load(ts_path, map_location=device))
    model_ts.to(device).eval()

    model_st = StaticOnlyModel()
    model_st.load_state_dict(torch.load(static_path, map_location=device))
    model_st.to(device).eval()

    X_train_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    X_train_ts     = np.asarray(dataset.X_train_ts[TS_NAME], dtype=np.float32)
    cache: Dict[Tuple[bytes, bytes], float] = {}
    lock  = threading.Lock()

    for s in range(0, len(X_train_static), 128):
        e  = min(s + 128, len(X_train_static))
        xt = torch.tensor(X_train_ts[s:e],     dtype=torch.float32, device=device)
        xs = torch.tensor(X_train_static[s:e], dtype=torch.float32, device=device)
        with torch.no_grad():
            p_ts  = model_ts(xt).squeeze(1).cpu().numpy()
            p_st  = model_st(xs).squeeze(1).cpu().numpy()
        p_avg = 0.5 * p_ts + 0.5 * p_st
        for i, p in enumerate(p_avg):
            key = (X_train_static[s+i].tobytes(), X_train_ts[s+i].tobytes())
            cache[key] = float(p >= 0.5)

    def _predict_fn(x_tab, x_ts, _text=None):
        xt  = _unwrap_ts(x_ts)
        xs  = np.asarray(x_tab, dtype=np.float32)
        hit = cache.get((xs.tobytes(), xt.tobytes()))
        if hit is not None:
            return hit
        with lock:
            with torch.no_grad():
                p_ts = float(model_ts(
                    torch.tensor(xt[None], dtype=torch.float32, device=device)
                ).squeeze().item())
                p_st = float(model_st(
                    torch.tensor(xs[None], dtype=torch.float32, device=device)
                ).squeeze().item())
        return float((0.5 * p_ts + 0.5 * p_st) >= 0.5)

    return _predict_fn, (model_ts, model_st)


def _build_predict_fn_late_nondp(dataset, rocket_path, static_path, device):
    """ROCKET (TS) + sklearn static model, averaged probability."""
    with open(rocket_path, "rb") as fh:
        rocket = pickle.load(fh)
    with open(static_path, "rb") as fh:
        static_model = pickle.load(fh)

    X_train_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    X_train_ts     = np.asarray(dataset.X_train_ts[TS_NAME], dtype=np.float32)
    cache: Dict[Tuple[bytes, bytes], float] = {}
    lock  = threading.Lock()

    X_tr_aeon = X_train_ts.transpose(0, 2, 1)
    p_ts_all  = rocket.predict_proba(X_tr_aeon)[:, 1].astype(np.float32)
    p_st_all  = static_model.predict_proba(X_train_static)[:, 1].astype(np.float32)
    for i in range(len(X_train_static)):
        key = (X_train_static[i].tobytes(), X_train_ts[i].tobytes())
        cache[key] = float((0.5 * p_ts_all[i] + 0.5 * p_st_all[i]) >= 0.5)

    def _predict_fn(x_tab, x_ts, _text=None):
        xt  = _unwrap_ts(x_ts)
        xs  = np.asarray(x_tab, dtype=np.float32)
        hit = cache.get((xs.tobytes(), xt.tobytes()))
        if hit is not None:
            return hit
        # aeon expects (1, C, T)
        xt_aeon = xt.T[np.newaxis]  # (1, 53, 24)
        with lock:
            p_ts = float(rocket.predict_proba(xt_aeon)[0, 1])
            p_st = float(static_model.predict_proba(xs[np.newaxis])[0, 1])
        return float((0.5 * p_ts + 0.5 * p_st) >= 0.5)

    return _predict_fn, None


# ===========================================================================
# Main loop
# ===========================================================================
tab_metrics = ["euclidean", "manhattan"]
ts_metrics  = ["dtw", "euclidean", "lcss"]
dtw_windows = [0.10]

for strategy, best_key, entry, run_name, missing_folds in todo:
    model_type = entry["model_type"]
    model_files = entry.get("model_files", {})

    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}  |  model: {best_key}  |  run: {run_name}")
    print(f"Folds to run: {missing_folds}")
    print(f"{'='*60}")

    for fold in missing_folds:
        fold_dir = DATA_DIR / f"fold_{fold}"
        print(f"\n--- Fold {fold} ---")

        # Build dataset for this fold
        produced = build_sepsis_dataset(fold=fold, gpu=args.gpu, load_model=False)
        dataset  = produced["dataset"]
        y_pred   = produced["y_pred"]

        if y_pred is not None:
            sample_indices = [int(i) for i, p in enumerate(y_pred) if int(p) == source_value]
        else:
            sample_indices = [int(i) for i in range(len(dataset.y_test))
                              if int(dataset.y_test[i]) == source_value]
        if args.max_samples is not None:
            sample_indices = sample_indices[:args.max_samples]

        if not sample_indices:
            print(f"  No source-class samples in fold {fold} — skipping.")
            continue

        print(f"  {len(sample_indices)} samples to explain")

        # Build predict_fn
        try:
            if model_type == "pytorch_early_fusion_sepsis":
                mp = model_files.get(f"fold_{fold}")
                if mp is None:
                    mp = str(DATA_DIR / f"best_model_{fold}.pt")
                predict_fn, torch_model = _build_predict_fn_early(dataset, mp, DEVICE)
            elif model_type == "pytorch_intermediate_fusion":
                mp = model_files.get(f"fold_{fold}")
                if mp is None:
                    mp = str(DATA_DIR / f"best_model_intermediate_fusion_mlp_fold{fold}.pt")
                predict_fn, torch_model = _build_predict_fn_if_mlp(dataset, mp, DEVICE)
            elif model_type == "sklearn_intermediate_fusion":
                mp     = model_files.get(f"fold_{fold}")
                mlp_mp = model_files.get(f"fold_{fold}_if_mlp")
                predict_fn, torch_model = _build_predict_fn_sklearn_if(
                    dataset, mp, mlp_mp, DEVICE
                )
            elif model_type == "pytorch_late_fusion_deep":
                ts_mp = model_files.get(f"fold_{fold}_ts",
                                         str(DATA_DIR / f"best_model_late_fusion_mlp_ts_fold{fold}.pt"))
                st_mp = model_files.get(f"fold_{fold}_static",
                                         str(DATA_DIR / f"best_model_late_fusion_mlp_static_fold{fold}.pt"))
                predict_fn, torch_model = _build_predict_fn_late_deep(
                    dataset, ts_mp, st_mp, DEVICE
                )
            elif model_type == "sklearn_late_fusion_nondp":
                r_mp  = model_files.get(f"fold_{fold}_rocket",
                                         str(DATA_DIR / f"best_model_late_fusion_rocket_fold{fold}.pkl"))
                st_mp = model_files.get(f"fold_{fold}_static",
                                         str(DATA_DIR / f"best_model_late_fusion_gbt_static_fold{fold}.pkl"))
                predict_fn, torch_model = _build_predict_fn_late_nondp(
                    dataset, r_mp, st_mp, DEVICE
                )
            else:
                print(f"  [warn] Unknown model_type={model_type} — skipping fold {fold}.")
                continue
        except FileNotFoundError as exc:
            print(f"  [warn] {exc} — skipping fold {fold}.")
            continue

        # Latents for IntermediateFusion generator
        train_latents, test_latents = None, None
        include_if = model_type == "pytorch_intermediate_fusion"
        if include_if and torch_model is not None:
            try:
                train_latents, test_latents = _compute_latents(
                    torch_model, dataset, DEVICE
                )
            except Exception as exc:
                print(f"  [warn] Could not extract latents: {exc}")
                include_if = False

        # Plausibility normalizer and tau_c
        plaus_norm = fit_plausibility_normalizer(
            X_tab_obs=dataset.X_train_static,
            X_ts_obs=dataset.X_train_ts[TS_NAME],
            y_train=dataset.y_train,
            target_value=target_value,
        )
        tau_c      = np.zeros(dataset.X_train_ts[TS_NAME].shape[-1], dtype=np.float32)
        ts_segments = [(0, int(dataset.X_train_ts[TS_NAME].shape[1]))]
        k_search    = min(50, args.k * 5)
        primary_tab = dataset.primary_tabular_name

        def _objectives_kwargs_factory(_txt_cfg, _img_cfg,
                                        _tab=primary_tab, _pn=plaus_norm,
                                        _ts=ts_segments, _tc=tau_c, _pfn=predict_fn):
            return {
                "y_target":   target_value,
                "predict_fn": _pfn,
                "tabular_objective_contexts": {
                    (_tab or "__primary__"): {
                        "plausibility_normalizer": {
                            "lof":  _pn.get("tab_lof"),
                            "low":  _pn.get("tab_low",  0.0),
                            "high": _pn.get("tab_high", 1.0),
                        }
                    }
                },
                "ts_objective_contexts": {
                    TS_NAME: {
                        "segments": _ts,
                        "tau_c":    _tc,
                        "rho":      0.0,
                        "plausibility_normalizer": {
                            "lof":  _pn.get("ts_lof"),
                            "low":  _pn.get("ts_low",  0.0),
                            "high": _pn.get("ts_high", 1.0),
                        },
                    }
                },
            }

        def _generators_factory(tab_cfg, ts_cfg, txt_cfg, txt_bk,
                                 img_cfg, img_bk,
                                 _k=args.k, _ks=k_search, _if=include_if,
                                 _tl=train_latents, _tel=test_latents,
                                 _pfn=predict_fn, _tab=primary_tab):
            del txt_cfg, txt_bk, img_cfg, img_bk
            tab_metric  = (tab_cfg or {}).get(_tab or "__primary__", "euclidean")
            static_dist = _metric_to_static_dist_fn(tab_metric)
            gens = {
                "Frankenstein": FrankensteinNN(k=_k, k_search=_ks,
                                               static_dist_fn=static_dist),
                "Combined":     CombinedNN(    k=_k, k_search=_ks,
                                               static_dist_fn=static_dist),
                "EarlyFusion":  EarlyFusionNN( k=_k, distance_metric=tab_metric),
            }
            if _if and _tl is not None:
                gens["IntermediateFusion"] = PyTorchLatentNN(
                    k=_k, train_latents=_tl, test_latents=_tel,
                    distance_metric=tab_metric,
                )
            return gens

        fold_output_dir = str(Path(args.output_dir) / f"fold_{fold}")
        run_distance_ablation(
            dataset=dataset,
            model=None,
            sample_indices=sample_indices,
            target_value=target_value,
            k=args.k,
            tab_metrics=tab_metrics,
            ts_metrics=ts_metrics,
            dtw_windows=dtw_windows,
            text_encoders=[],
            image_encoders=[],
            output_dir=fold_output_dir,
            run_name=run_name,
            save_full=args.save_full,
            max_combinations=None,
            n_jobs=args.n_jobs,
            objectives_kwargs_factory=_objectives_kwargs_factory,
            extra_generators_factory=_generators_factory,
        )
        print(f"  Done — {fold_output_dir}/{run_name}/summary.json")

print("\nAll requested ablations complete.")
