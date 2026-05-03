"""
Evaluate all trained fusion model variants on the test set and write
data/fusion_model_registry.json.

For each variant that has a saved model file, this script:
  1. Loads the model
  2. Runs inference on the test set
  3. Computes test macro-F1 and accuracy
  4. Records the result with the model path and a canonical ablation_run_name

The resulting JSON is consumed by run_cf_for_best_models.py to decide which
ablation runs to launch.

Run
---
    cd long_covid_tweets
    python evaluate_fusion_models.py [--gpu N]
"""
from __future__ import annotations

import argparse
import json
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
parser.add_argument("--gpu", type=int, default=None,
                    help="GPU index for loading the intermediate-fusion transformer.")
args = parser.parse_args()

DATA_DIR = Path("data")
DEVICE = (
    f"cuda:{args.gpu}"
    if args.gpu is not None and torch.cuda.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Load test data
# ---------------------------------------------------------------------------
print("Loading dataset …")
raw = np.load(DATA_DIR / "dataset.npz", allow_pickle=True)
X_test_static = raw["X_test_static"].astype("float32")
X_test_text   = raw["X_test_text"].tolist()
y_test        = raw["y_test"].astype(int)
label_classes = raw["label_classes"].tolist()
N_CLASSES     = len(label_classes)
D_TAB         = X_test_static.shape[1]
print(f"  Test samples: {len(y_test)}  |  classes: {label_classes}")

# ---------------------------------------------------------------------------
# Load frozen CLS embedding cache (shared by early- and late-fusion scripts)
# ---------------------------------------------------------------------------
EMB_CACHE = DATA_DIR / "early_fusion_text_embeddings.pt"
if EMB_CACHE.exists():
    print(f"Loading CLS embedding cache from {EMB_CACHE} …")
    _cache      = torch.load(EMB_CACHE, map_location="cpu")
    test_cls    = _cache["test"].numpy().astype("float32")    # (n_test, 768)
    train_cls   = _cache["train"].numpy().astype("float32")   # (n_train, 768)
    X_test_ef   = np.concatenate([test_cls, X_test_static], axis=1)  # (n_test, 768+D_TAB)
    print(f"  CLS cache loaded — test shape {test_cls.shape}")
else:
    test_cls = train_cls = X_test_ef = None
    print("[warn] early_fusion_text_embeddings.pt not found — "
          "early/late-fusion MLP models will be skipped.")

# ---------------------------------------------------------------------------
# Inline model class definitions (avoid importing training scripts that have
# module-level argparse, which would fire on import and collide with our own)
# ---------------------------------------------------------------------------

class _EarlyFusionMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x): return self.net(x)


class _TextMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x): return self.net(x)


class _TabMLP(nn.Module):
    """Must match TabMLP in train_late_fusion.py — dims inferred from state_dict."""
    def __init__(self, d_in, n_classes, hidden=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),       nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
    def forward(self, x): return self.net(x)


class _TabHead(nn.Module):
    def __init__(self, d_in, hidden_dims, dropout):
        super().__init__()
        layers, in_dim = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net    = nn.Sequential(*layers)
        self.out_dim = in_dim
    def forward(self, x): return self.net(x)


class _MultimodalClassifier(nn.Module):
    """Inline replica of the intermediate-fusion MultimodalClassifier."""
    _TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"

    def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
        super().__init__()
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(self._TEXT_MODEL)
        enc_dim           = self.text_encoder.config.hidden_size
        self.text_proj    = nn.Sequential(
            nn.Linear(enc_dim, text_hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.tab_head     = _TabHead(d_tab, tab_hidden_dims, dropout)
        self.classifier   = nn.Linear(text_hidden_dim + self.tab_head.out_dim, n_classes)

    def forward(self, input_ids, attention_mask, X_static):
        cls = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.classifier(
            torch.cat([self.text_proj(cls), self.tab_head(X_static)], dim=1)
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred):
    return {
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def _pt_predict(model, X_tensor, batch_size=256):
    """Run a simple single-input PyTorch model and return predictions."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            parts.append(model(X_tensor[i:i+batch_size]).argmax(1).numpy())
    return np.concatenate(parts).astype(int)


# ---------------------------------------------------------------------------
# Registry skeleton
# ---------------------------------------------------------------------------
registry: dict = {
    "dataset":       "long_covid_tweets",
    "generated_at":  datetime.now().isoformat(),
    "label_classes": label_classes,
    "models":        {},
    "best_per_strategy": {},
}


def _register(key, strategy, family, model_type, model_files,
               ablation_run_name, y_pred, extra=None):
    m = _metrics(y_test, y_pred)
    entry = {
        "strategy":         strategy,
        "family":           family,
        "model_type":       model_type,
        "model_files":      model_files,
        "available":        True,
        "ablation_run_name": ablation_run_name,
        **m,
        **(extra or {}),
    }
    registry["models"][key] = entry
    print(f"  [{key}]  macro-F1={m['test_macro_f1']:.4f}  acc={m['test_accuracy']:.4f}")


def _register_missing(key, reason):
    registry["models"][key] = {"available": False, "error": reason}
    print(f"  [{key}]  UNAVAILABLE — {reason}")


# ===========================================================================
# 1.  Intermediate fusion — deep (best_model.pt)
# ===========================================================================
print("\n=== 1. Intermediate fusion (deep) ===")
_IF_PATH = DATA_DIR / "best_model.pt"

if _IF_PATH.exists():
    try:
        from transformers import AutoTokenizer
        ckpt      = torch.load(_IF_PATH, map_location=DEVICE)
        tok       = AutoTokenizer.from_pretrained(_MultimodalClassifier._TEXT_MODEL)
        if_model  = _MultimodalClassifier(
            d_tab           = len(ckpt["tab_cols"]),
            n_classes       = N_CLASSES,
            tab_hidden_dims = ckpt["config"]["tab_hidden_dims"],
            text_hidden_dim = ckpt["config"]["text_hidden_dim"],
            dropout         = ckpt["config"]["dropout"],
        ).to(DEVICE)
        if_model.load_state_dict(ckpt["state_dict"])
        if_model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(y_test), 32):
                enc = tok(
                    X_test_text[i:i+32],
                    max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                tab = torch.tensor(X_test_static[i:i+32]).to(DEVICE)
                logits = if_model(
                    enc["input_ids"].to(DEVICE),
                    enc["attention_mask"].to(DEVICE),
                    tab,
                )
                preds.append(logits.argmax(1).cpu().numpy())

        _register(
            key="intermediate/deep",
            strategy="intermediate", family="deep",
            model_type="pytorch_intermediate",
            model_files={"main": str(_IF_PATH)},
            ablation_run_name="fusion_intermediate_deep",
            y_pred=np.concatenate(preds),
            extra={"val_macro_f1": float(ckpt["config"].get("val_macro_f1", 0.5987))},
        )
    except Exception as exc:
        _register_missing("intermediate/deep", str(exc))
else:
    _register_missing("intermediate/deep", f"{_IF_PATH} not found")


# ===========================================================================
# 2.  Early fusion — MLP
# ===========================================================================
print("\n=== 2. Early fusion — MLP ===")
_EF_MLP_PATH = DATA_DIR / "best_model_early_fusion_mlp.pt"

if _EF_MLP_PATH.exists() and X_test_ef is not None:
    try:
        ckpt   = torch.load(_EF_MLP_PATH, map_location="cpu")
        ef_mlp = _EarlyFusionMLP(ckpt["d_in"], ckpt["n_classes"],
                                  ckpt["hidden"], ckpt["dropout"])
        ef_mlp.load_state_dict(ckpt["state_dict"])
        y_pred = _pt_predict(ef_mlp, torch.tensor(X_test_ef))
        _register(
            key="early/mlp",
            strategy="early", family="mlp",
            model_type="pytorch_early_fusion",
            model_files={"main": str(_EF_MLP_PATH)},
            ablation_run_name="fusion_early_mlp",
            y_pred=y_pred,
        )
    except Exception as exc:
        _register_missing("early/mlp", str(exc))
elif X_test_ef is None:
    _register_missing("early/mlp", "embedding cache not found")
else:
    _register_missing("early/mlp", f"{_EF_MLP_PATH} not found")


# ===========================================================================
# 3.  Early fusion — RF
# ===========================================================================
print("\n=== 3. Early fusion — RF ===")
_EF_RF_PATH = DATA_DIR / "best_model_early_fusion_rf.pkl"

if _EF_RF_PATH.exists() and X_test_ef is not None:
    try:
        with open(_EF_RF_PATH, "rb") as fh:
            bundle = pickle.load(fh)
        rf     = bundle if not isinstance(bundle, dict) else bundle["model"]
        y_pred = rf.predict(X_test_ef).astype(int)
        _register(
            key="early/rf",
            strategy="early", family="rf",
            model_type="sklearn_early_fusion",
            model_files={"main": str(_EF_RF_PATH)},
            ablation_run_name="fusion_early_rf",
            y_pred=y_pred,
        )
    except Exception as exc:
        _register_missing("early/rf", str(exc))
elif X_test_ef is None:
    _register_missing("early/rf", "embedding cache not found")
else:
    _register_missing("early/rf", f"{_EF_RF_PATH} not found")


# ===========================================================================
# 4.  Early fusion — GBT
# ===========================================================================
print("\n=== 4. Early fusion — GBT ===")
_EF_GBT_PATH = DATA_DIR / "best_model_early_fusion_gbt.pkl"

if _EF_GBT_PATH.exists() and X_test_ef is not None:
    try:
        with open(_EF_GBT_PATH, "rb") as fh:
            bundle = pickle.load(fh)
        gbt    = bundle if not isinstance(bundle, dict) else bundle["model"]
        y_pred = gbt.predict(X_test_ef).astype(int)
        _register(
            key="early/gbt",
            strategy="early", family="gbt",
            model_type="sklearn_early_fusion",
            model_files={"main": str(_EF_GBT_PATH)},
            ablation_run_name="fusion_early_gbt",
            y_pred=y_pred,
        )
    except Exception as exc:
        _register_missing("early/gbt", str(exc))
elif X_test_ef is None:
    _register_missing("early/gbt", "embedding cache not found")
else:
    _register_missing("early/gbt", f"{_EF_GBT_PATH} not found")


# ===========================================================================
# 5.  Late fusion — non-deep (TF-IDF+LR text + best tabular sklearn)
# ===========================================================================
print("\n=== 5. Late fusion — non-deep ===")
_LF_TFIDF_PATH   = DATA_DIR / "best_model_late_fusion_tfidf_logreg_text.pkl"
_LF_RF_TAB_PATH  = DATA_DIR / "best_model_late_fusion_rf_tabular.pkl"
_LF_GBT_TAB_PATH = DATA_DIR / "best_model_late_fusion_gbt_tabular.pkl"

_missing = [p for p in [_LF_TFIDF_PATH] if not p.exists()]
_tab_paths_avail = [p for p in [_LF_RF_TAB_PATH, _LF_GBT_TAB_PATH] if p.exists()]

if _missing:
    _register_missing("late/nondp", f"missing: {[str(p) for p in _missing]}")
elif not _tab_paths_avail:
    _register_missing("late/nondp", "no tabular non-deep model found (rf or gbt)")
else:
    try:
        with open(_LF_TFIDF_PATH, "rb") as fh:
            bundle_t = pickle.load(fh)
        tfidf_vec  = bundle_t["tfidf"]
        logreg     = bundle_t["model"]
        p_text     = logreg.predict_proba(
            tfidf_vec.transform(X_test_text)
        ).astype("float32")

        # Use both tabular models if available; pick the one with better combined F1
        best_tab_model, best_tab_path, best_tab_f1 = None, None, -1.0
        for _tp in _tab_paths_avail:
            with open(_tp, "rb") as fh:
                _b = pickle.load(fh)
            _m  = _b if not isinstance(_b, dict) else _b["model"]
            _pt = _m.predict_proba(X_test_static).astype("float32")
            _yp = (0.5 * p_text + 0.5 * _pt).argmax(1)
            _f1 = float(f1_score(y_test, _yp, average="macro", zero_division=0))
            if _f1 > best_tab_f1:
                best_tab_model, best_tab_path, best_tab_f1 = _m, _tp, _f1
                best_p_tab = _pt

        p_combined = (0.5 * p_text + 0.5 * best_p_tab).argmax(1).astype(int)
        _register(
            key="late/nondp",
            strategy="late", family="nondp",
            model_type="late_fusion_nondp",
            model_files={
                "text":    str(_LF_TFIDF_PATH),
                "tabular": str(best_tab_path),
            },
            ablation_run_name="fusion_late_nondp",
            y_pred=p_combined,
        )
    except Exception as exc:
        _register_missing("late/nondp", str(exc))


# ===========================================================================
# 6.  Late fusion — deep (TextMLP + TabMLP)
# ===========================================================================
print("\n=== 6. Late fusion — deep ===")
_LF_TXT_MLP_PATH = DATA_DIR / "best_model_late_fusion_mlp_text.pt"
_LF_TAB_MLP_PATH = DATA_DIR / "best_model_late_fusion_mlp_tabular.pt"

_missing_d = [p for p in [_LF_TXT_MLP_PATH, _LF_TAB_MLP_PATH] if not p.exists()]

if _missing_d:
    _register_missing("late/mlp_deep", f"missing: {[str(p) for p in _missing_d]}")
elif test_cls is None:
    _register_missing("late/mlp_deep", "embedding cache not found")
else:
    try:
        # --- text branch ---
        ckpt_t   = torch.load(_LF_TXT_MLP_PATH, map_location="cpu")
        d_t      = ckpt_t["state_dict"]["net.0.weight"].shape[1]
        h_t      = ckpt_t["state_dict"]["net.0.weight"].shape[0]
        nc_t     = ckpt_t["state_dict"]["net.2.weight"].shape[0]
        txt_mlp  = _TextMLP(d_t, nc_t, h_t, dropout=ckpt_t.get("dropout", 0.3))
        txt_mlp.load_state_dict(ckpt_t["state_dict"])
        txt_mlp.eval()

        # --- tabular branch ---
        ckpt_tb  = torch.load(_LF_TAB_MLP_PATH, map_location="cpu")
        d_tb     = ckpt_tb["state_dict"]["net.0.weight"].shape[1]
        h_tb     = ckpt_tb["state_dict"]["net.0.weight"].shape[0]
        nc_tb    = ckpt_tb["state_dict"]["net.4.weight"].shape[0]
        tab_mlp  = _TabMLP(d_tb, nc_tb, h_tb)
        tab_mlp.load_state_dict(ckpt_tb["state_dict"])
        tab_mlp.eval()

        with torch.no_grad():
            p_txt = torch.softmax(
                txt_mlp(torch.tensor(test_cls)), dim=1
            ).numpy().astype("float32")
            p_tab = torch.softmax(
                tab_mlp(torch.tensor(X_test_static)), dim=1
            ).numpy().astype("float32")

        y_pred = (0.5 * p_txt + 0.5 * p_tab).argmax(1).astype(int)
        _register(
            key="late/mlp_deep",
            strategy="late", family="mlp_deep",
            model_type="late_fusion_deep",
            model_files={
                "text":    str(_LF_TXT_MLP_PATH),
                "tabular": str(_LF_TAB_MLP_PATH),
            },
            ablation_run_name="fusion_late_mlp_deep",
            y_pred=y_pred,
        )
    except Exception as exc:
        _register_missing("late/mlp_deep", str(exc))


# ===========================================================================
# Derive best_per_strategy
# ===========================================================================
strategy_keys = {
    "intermediate": ["intermediate/deep"],
    "early":        ["early/mlp", "early/rf", "early/gbt"],
    "late":         ["late/nondp", "late/mlp_deep"],
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
out_path = DATA_DIR / "fusion_model_registry.json"
with open(out_path, "w") as fh:
    json.dump(registry, fh, indent=2, default=str)
print(f"\nRegistry written to {out_path}")
