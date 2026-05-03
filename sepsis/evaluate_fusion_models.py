"""
Evaluate all trained fusion model variants on the test set (pooled across folds)
and write sepsis/data/fusion_model_registry.json.

Fusion variants
---------------
  1. Early fusion  (existing SepsisModel from train_pytorch.py)
       Best model per fold  : data/best_model_{fold}.pt
       Precomputed y_pred   : data/fold_{fold}/y_pred.npy  (written by evaluate.py)
       Falls back to live model inference when y_pred.npy is absent.

  2. Intermediate fusion  (SepsisIFModel from train_intermediate_fusion.py)
       deep MLP : data/best_model_intermediate_fusion_mlp_fold{N}.pt
       RF       : data/best_model_intermediate_fusion_rf_fold{N}.pkl
       GBT      : data/best_model_intermediate_fusion_gbt_fold{N}.pkl

  3. Late fusion  (from train_late_fusion.py)
       deep  : data/best_model_late_fusion_mlp_ts_fold{N}.pt
                + data/best_model_late_fusion_mlp_static_fold{N}.pt
       nondp : data/best_model_late_fusion_rocket_fold{N}.pkl
                + data/best_model_late_fusion_gbt_static_fold{N}.pkl

For each model type, metrics are pooled across all available folds and reported
as test_macro_f1 / test_accuracy.

Run
---
    python sepsis/evaluate_fusion_models.py [--gpu N]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",     type=int, default=None)
parser.add_argument("--n-folds", type=int, default=5)
args = parser.parse_args()

DATA_DIR = Path(__file__).resolve().parent / "data"
N_FOLDS  = args.n_folds
LABEL_CLASSES = ["no_death", "death"]
N_CLASSES     = len(LABEL_CLASSES)

DEVICE = (
    f"cuda:{args.gpu}"
    if args.gpu is not None and torch.cuda.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Inline model definitions — must match the training scripts exactly
# ---------------------------------------------------------------------------

class SepsisIFModel(nn.Module):
    """Intermediate-fusion model (train_intermediate_fusion.py)."""
    def __init__(self, n_ts_features=53, n_static_features=47,
                 ts_hidden=32, static_hidden=32, fused_hidden=32, dropout_rate=0.2):
        super().__init__()
        self.gru      = nn.GRU(input_size=n_ts_features, hidden_size=ts_hidden,
                               batch_first=True)
        self.ts_drop  = nn.Dropout(dropout_rate)
        self.static_enc = nn.Sequential(
            nn.Linear(n_static_features, static_hidden), nn.ReLU(), nn.Dropout(dropout_rate),
        )
        fusion_in = ts_hidden + static_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fused_hidden), nn.Tanh(),
            nn.BatchNorm1d(fused_hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Linear(fused_hidden, 1)

    def encode(self, x_ts, x_static):
        _, h = self.gru(x_ts)
        ts_emb     = self.ts_drop(h.squeeze(0))
        static_emb = self.static_enc(x_static)
        return self.fusion(torch.cat([ts_emb, static_emb], dim=1))

    def forward(self, x_ts, x_static):
        return torch.sigmoid(self.output_layer(self.encode(x_ts, x_static)))


class TSOnlyModel(nn.Module):
    """GRU TS branch (train_late_fusion.py)."""
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
        h = torch.tanh(self.dense(h))
        h = self.bn(h)
        return torch.sigmoid(self.out(h))


class StaticOnlyModel(nn.Module):
    """MLP static branch (train_late_fusion.py)."""
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
# Registry helpers
# ---------------------------------------------------------------------------
registry: dict = {
    "dataset":           "sepsis",
    "generated_at":      datetime.now().isoformat(),
    "label_classes":     LABEL_CLASSES,
    "n_folds":           N_FOLDS,
    "models":            {},
    "best_per_strategy": {},
}


def _metrics(y_true, y_pred):
    return {
        "test_macro_f1":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def _register(key, strategy, family, model_type, model_files,
               ablation_run_name, y_true_pool, y_pred_pool, extra=None):
    m = _metrics(y_true_pool, y_pred_pool)
    registry["models"][key] = {
        "strategy": strategy, "family": family, "model_type": model_type,
        "model_files": model_files, "available": True,
        "ablation_run_name": ablation_run_name,
        **m, **(extra or {}),
    }
    print(f"  [{key}]  macro-F1={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")


def _register_missing(key, reason):
    registry["models"][key] = {"available": False, "error": reason}
    print(f"  [{key}]  UNAVAILABLE — {reason}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predict_proba_torch(model, X_ts, X_static, device, batch_size=256):
    model.eval()
    out = []
    for s in range(0, len(X_ts), batch_size):
        e  = min(s + batch_size, len(X_ts))
        xt = torch.tensor(X_ts[s:e],     dtype=torch.float32, device=device)
        xs = torch.tensor(X_static[s:e], dtype=torch.float32, device=device)
        out.append(model(xt, xs).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def _predict_proba_ts_only(model, X_ts, device, batch_size=256):
    model.eval()
    out = []
    for s in range(0, len(X_ts), batch_size):
        e  = min(s + batch_size, len(X_ts))
        xt = torch.tensor(X_ts[s:e], dtype=torch.float32, device=device)
        out.append(model(xt).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def _predict_proba_static_only(model, X_static, device, batch_size=256):
    model.eval()
    out = []
    for s in range(0, len(X_static), batch_size):
        e  = min(s + batch_size, len(X_static))
        xs = torch.tensor(X_static[s:e], dtype=torch.float32, device=device)
        out.append(model(xs).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


@torch.no_grad()
def _extract_if_latents(model, X_ts, X_static, device, batch_size=256):
    model.eval()
    out = []
    for s in range(0, len(X_ts), batch_size):
        e  = min(s + batch_size, len(X_ts))
        xt = torch.tensor(X_ts[s:e],     dtype=torch.float32, device=device)
        xs = torch.tensor(X_static[s:e], dtype=torch.float32, device=device)
        out.append(model.encode(xt, xs).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ===========================================================================
# 1.  Early fusion — existing SepsisModel
# ===========================================================================
print("\n=== 1. Early fusion (existing SepsisModel) ===")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_pytorch import SepsisModel as _SepsisEarlyModel  # noqa: E402
except ImportError as _e:
    _SepsisEarlyModel = None
    print(f"  [warn] Could not import SepsisModel: {_e}")

_ef_true, _ef_pred, _ef_model_files = [], [], {}
_ef_missing_folds = []

for fold in range(N_FOLDS):
    fold_dir  = DATA_DIR / f"fold_{fold}"
    y_pred_path = fold_dir / "y_pred.npy"
    y_test_path = fold_dir / "y_test.npy"

    if not y_test_path.exists():
        print(f"  fold {fold}: y_test.npy missing — skipping.")
        continue

    y_test = np.load(y_test_path).astype(int)

    if y_pred_path.exists():
        y_pred = np.load(y_pred_path).astype(int)
        _ef_true.append(y_test)
        _ef_pred.append(y_pred)
        _ef_model_files[f"fold_{fold}"] = str(DATA_DIR / f"best_model_{fold}.pt")
    else:
        # Fall back to live model inference
        model_path = DATA_DIR / f"best_model_{fold}.pt"
        if not model_path.exists() or _SepsisEarlyModel is None:
            _ef_missing_folds.append(fold)
            print(f"  fold {fold}: y_pred.npy and model not found — skipping.")
            continue
        X_test_ts     = np.load(fold_dir / "X_test_ts.npy").astype("float32")
        X_test_static = np.load(fold_dir / "X_test_static.npy").astype("float32")
        model_ef = _SepsisEarlyModel()
        model_ef.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model_ef.to(DEVICE).eval()
        proba  = _predict_proba_torch(model_ef, X_test_ts, X_test_static, DEVICE)
        y_pred = (proba >= 0.5).astype(int)
        _ef_true.append(y_test)
        _ef_pred.append(y_pred)
        _ef_model_files[f"fold_{fold}"] = str(model_path)
        del model_ef

if _ef_true:
    _register(
        key="early/deep",
        strategy="early", family="deep",
        model_type="pytorch_early_fusion_sepsis",
        model_files=_ef_model_files,
        ablation_run_name="fusion_early_deep",
        y_true_pool=np.concatenate(_ef_true),
        y_pred_pool=np.concatenate(_ef_pred),
    )
else:
    _register_missing("early/deep",
                      f"no early-fusion fold data found (missing folds: {_ef_missing_folds})")


# ===========================================================================
# 2–4.  Intermediate fusion (MLP / RF / GBT)
# ===========================================================================
print("\n=== 2–4. Intermediate fusion ===")

def _eval_intermediate(key, family, model_type, model_file_tpl,
                        label, if_mlp_file_tpl=None):
    """Evaluate an intermediate-fusion model family pooled across folds."""
    all_true, all_pred, model_files = [], [], {}

    for fold in range(N_FOLDS):
        fold_dir      = DATA_DIR / f"fold_{fold}"
        y_test_path   = fold_dir / "y_test.npy"
        if not y_test_path.exists():
            continue

        y_test        = np.load(y_test_path).astype(int)
        X_test_ts     = np.load(fold_dir / "X_test_ts.npy").astype("float32")
        X_test_static = np.load(fold_dir / "X_test_static.npy").astype("float32")
        m_path        = DATA_DIR / model_file_tpl.format(fold=fold)

        if not m_path.exists():
            print(f"  [{key}] fold {fold}: {m_path.name} not found — skipping.")
            continue

        try:
            if model_type == "pytorch_intermediate_fusion":
                model = SepsisIFModel().to(DEVICE)
                model.load_state_dict(torch.load(m_path, map_location=DEVICE))
                proba  = _predict_proba_torch(model, X_test_ts, X_test_static, DEVICE)
                y_pred = (proba >= 0.5).astype(int)
                del model
            else:
                # RF or GBT on latents — need the IF-MLP model to extract latents
                mlp_path = DATA_DIR / if_mlp_file_tpl.format(fold=fold)
                if not mlp_path.exists():
                    print(f"  [{key}] fold {fold}: IF-MLP {mlp_path.name} not found — skipping.")
                    continue
                mlp_model = SepsisIFModel().to(DEVICE)
                mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
                Z_test = _extract_if_latents(mlp_model, X_test_ts, X_test_static, DEVICE)
                del mlp_model

                with open(m_path, "rb") as fh:
                    sk_model = pickle.load(fh)
                y_pred = sk_model.predict(Z_test).astype(int)

            all_true.append(y_test)
            all_pred.append(y_pred)
            model_files[f"fold_{fold}"] = str(m_path)
            if if_mlp_file_tpl:
                model_files[f"fold_{fold}_if_mlp"] = str(
                    DATA_DIR / if_mlp_file_tpl.format(fold=fold)
                )

        except Exception as exc:
            print(f"  [{key}] fold {fold}: error — {exc}")

    if all_true:
        _register(
            key=key, strategy="intermediate", family=family,
            model_type=model_type,
            model_files=model_files,
            ablation_run_name=f"fusion_intermediate_{family}",
            y_true_pool=np.concatenate(all_true),
            y_pred_pool=np.concatenate(all_pred),
        )
    else:
        _register_missing(key, f"no {label} model files found")


_IF_MLP_TPL = "best_model_intermediate_fusion_mlp_fold{fold}.pt"

_eval_intermediate(
    "intermediate/mlp", "mlp", "pytorch_intermediate_fusion",
    "best_model_intermediate_fusion_mlp_fold{fold}.pt",
    "IF deep MLP",
)
_eval_intermediate(
    "intermediate/rf", "rf", "sklearn_intermediate_fusion",
    "best_model_intermediate_fusion_rf_fold{fold}.pkl",
    "IF RF",
    if_mlp_file_tpl=_IF_MLP_TPL,
)
_eval_intermediate(
    "intermediate/gbt", "gbt", "sklearn_intermediate_fusion",
    "best_model_intermediate_fusion_gbt_fold{fold}.pkl",
    "IF GBT",
    if_mlp_file_tpl=_IF_MLP_TPL,
)


# ===========================================================================
# 5.  Late fusion — deep (TSOnlyModel + StaticOnlyModel)
# ===========================================================================
print("\n=== 5. Late fusion — deep ===")
_ld_true, _ld_pred, _ld_files = [], [], {}

for fold in range(N_FOLDS):
    fold_dir      = DATA_DIR / f"fold_{fold}"
    y_test_path   = fold_dir / "y_test.npy"
    if not y_test_path.exists():
        continue

    ts_path = DATA_DIR / f"best_model_late_fusion_mlp_ts_fold{fold}.pt"
    st_path = DATA_DIR / f"best_model_late_fusion_mlp_static_fold{fold}.pt"

    if not ts_path.exists() or not st_path.exists():
        missing = [str(p) for p in [ts_path, st_path] if not p.exists()]
        print(f"  fold {fold}: missing {missing} — skipping.")
        continue

    try:
        y_test        = np.load(y_test_path).astype(int)
        X_test_ts     = np.load(fold_dir / "X_test_ts.npy").astype("float32")
        X_test_static = np.load(fold_dir / "X_test_static.npy").astype("float32")

        model_ts = TSOnlyModel().to(DEVICE)
        model_ts.load_state_dict(torch.load(ts_path, map_location=DEVICE))
        model_st = StaticOnlyModel().to(DEVICE)
        model_st.load_state_dict(torch.load(st_path, map_location=DEVICE))

        p_ts = _predict_proba_ts_only(model_ts, X_test_ts, DEVICE)
        p_st = _predict_proba_static_only(model_st, X_test_static, DEVICE)
        p_avg  = 0.5 * p_ts + 0.5 * p_st
        y_pred = (p_avg >= 0.5).astype(int)

        _ld_true.append(y_test)
        _ld_pred.append(y_pred)
        _ld_files[f"fold_{fold}_ts"]     = str(ts_path)
        _ld_files[f"fold_{fold}_static"] = str(st_path)
        del model_ts, model_st

    except Exception as exc:
        print(f"  fold {fold}: error — {exc}")

if _ld_true:
    _register(
        key="late/deep",
        strategy="late", family="deep",
        model_type="pytorch_late_fusion_deep",
        model_files=_ld_files,
        ablation_run_name="fusion_late_deep",
        y_true_pool=np.concatenate(_ld_true),
        y_pred_pool=np.concatenate(_ld_pred),
    )
else:
    _register_missing("late/deep", "no deep late-fusion model files found")


# ===========================================================================
# 6.  Late fusion — non-deep (ROCKET + best static sklearn)
# ===========================================================================
print("\n=== 6. Late fusion — non-deep ===")
_ln_true, _ln_pred, _ln_files = [], [], {}

for fold in range(N_FOLDS):
    fold_dir        = DATA_DIR / f"fold_{fold}"
    y_test_path     = fold_dir / "y_test.npy"
    if not y_test_path.exists():
        continue

    rocket_path = DATA_DIR / f"best_model_late_fusion_rocket_fold{fold}.pkl"
    static_path = DATA_DIR / f"best_model_late_fusion_gbt_static_fold{fold}.pkl"

    if not rocket_path.exists() or not static_path.exists():
        missing = [str(p) for p in [rocket_path, static_path] if not p.exists()]
        print(f"  fold {fold}: missing {missing} — skipping.")
        continue

    try:
        y_test        = np.load(y_test_path).astype(int)
        X_test_ts     = np.load(fold_dir / "X_test_ts.npy").astype("float32")
        X_test_static = np.load(fold_dir / "X_test_static.npy").astype("float32")

        with open(rocket_path, "rb") as fh:
            rocket = pickle.load(fh)
        with open(static_path, "rb") as fh:
            static_model = pickle.load(fh)

        # aeon expects (n_cases, n_channels, n_timepoints)
        X_test_ts_aeon = X_test_ts.transpose(0, 2, 1)
        p_ts     = rocket.predict_proba(X_test_ts_aeon)[:, 1].astype(np.float32)
        p_static = static_model.predict_proba(X_test_static)[:, 1].astype(np.float32)
        p_avg    = 0.5 * p_ts + 0.5 * p_static
        y_pred   = (p_avg >= 0.5).astype(int)

        _ln_true.append(y_test)
        _ln_pred.append(y_pred)
        _ln_files[f"fold_{fold}_rocket"] = str(rocket_path)
        _ln_files[f"fold_{fold}_static"] = str(static_path)

    except Exception as exc:
        print(f"  fold {fold}: error — {exc}")

if _ln_true:
    _register(
        key="late/nondp",
        strategy="late", family="nondp",
        model_type="sklearn_late_fusion_nondp",
        model_files=_ln_files,
        ablation_run_name="fusion_late_nondp",
        y_true_pool=np.concatenate(_ln_true),
        y_pred_pool=np.concatenate(_ln_pred),
    )
else:
    _register_missing("late/nondp", "no non-deep late-fusion model files found")


# ===========================================================================
# Derive best_per_strategy
# ===========================================================================
strategy_keys = {
    "early":        ["early/deep"],
    "intermediate": ["intermediate/mlp", "intermediate/rf", "intermediate/gbt"],
    "late":         ["late/deep", "late/nondp"],
}

print("\n=== Best model per strategy ===")
for strategy, keys in strategy_keys.items():
    candidates = [
        (k, registry["models"][k]["test_macro_f1"])
        for k in keys
        if registry["models"].get(k, {}).get("available", False)
    ]
    if candidates:
        best_key = max(candidates, key=lambda x: x[1])[0]
        registry["best_per_strategy"][strategy] = best_key
        print(f"  {strategy}: {best_key}  "
              f"(F1={registry['models'][best_key]['test_macro_f1']:.4f})")
    else:
        registry["best_per_strategy"][strategy] = None
        print(f"  {strategy}: NO AVAILABLE MODEL")


# ===========================================================================
# Write registry
# ===========================================================================
import json
out_path = DATA_DIR / "fusion_model_registry.json"
with open(out_path, "w") as fh:
    json.dump(registry, fh, indent=2, default=str)
print(f"\nRegistry written to {out_path}")
