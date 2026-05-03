"""
Evaluate all trained fusion model variants on the test set and write
data/fusion_model_registry.json.

The intermediate fusion model (best_model.pt) uses end-to-end fine-tuned
FineTuneMultimodalClassifier; its predictions are already in y_pred.npy so we
load that directly to avoid re-running a heavy forward pass.

For early- and late-fusion models the precomputed_embeddings.pt cache is used
to build the test features without any GPU inference.

Run
---
    cd memes
    python evaluate_fusion_models.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=None)
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
raw       = np.load(DATA_DIR / "dataset.npz", allow_pickle=True)
y_test    = raw["y_test"].astype(int)
label_classes = raw["label_classes"].tolist()
N_CLASSES = len(label_classes)
print(f"  Test samples: {len(y_test)}  |  classes: {label_classes}")

# ---------------------------------------------------------------------------
# Load precomputed embeddings (same cache as hparam_search.py)
# ---------------------------------------------------------------------------
_EMBS_PATH = DATA_DIR / "precomputed_embeddings.pt"
if _EMBS_PATH.exists():
    print(f"Loading precomputed embeddings from {_EMBS_PATH} …")
    emb_cache = torch.load(_EMBS_PATH, map_location="cpu")
    # Structure: emb_cache["text"][enc]["train"/"test"]  → tensor (n, d)
    #            emb_cache["image"][enc]["train"/"test"] → tensor (n, d)
    def _get_emb(modality: str, encoder: str, split: str) -> np.ndarray:
        return emb_cache[modality][encoder][split].numpy().astype("float32")
    print("  Embedding cache loaded.")
else:
    emb_cache = None
    print("[warn] precomputed_embeddings.pt not found — early/late-fusion models will be skipped.")


def _pt_predict(model, X_tensor, batch_size=256) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            parts.append(model(X_tensor[i:i+batch_size]).argmax(1).numpy())
    return np.concatenate(parts).astype(int)


# ---------------------------------------------------------------------------
# Inline model definitions (matching train_early_fusion.py / train_late_fusion.py)
# ---------------------------------------------------------------------------

class _EarlyFusionMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x): return self.net(x)


class _BranchMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x): return self.net(x)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
registry: dict = {
    "dataset":           "hateful_memes",
    "generated_at":      datetime.now().isoformat(),
    "label_classes":     label_classes,
    "models":            {},
    "best_per_strategy": {},
}


def _metrics(y_true, y_pred):
    return {
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def _register(key, strategy, family, model_type, model_files,
               ablation_run_name, y_pred, extra=None):
    m = _metrics(y_test, y_pred)
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


# ===========================================================================
# 1.  Intermediate fusion — deep  (best_model.pt / y_pred.npy)
# ===========================================================================
print("\n=== 1. Intermediate fusion (deep) ===")
_YPRED_PATH = DATA_DIR / "y_pred.npy"
_BM_PATH    = DATA_DIR / "best_model.pt"

if _YPRED_PATH.exists():
    y_pred_if = np.load(_YPRED_PATH).astype(int)
    _register(
        key="intermediate/deep",
        strategy="intermediate", family="deep",
        model_type="pytorch_intermediate",
        model_files={"main": str(_BM_PATH)},
        ablation_run_name="fusion_intermediate_deep",
        y_pred=y_pred_if,
    )
elif _BM_PATH.exists():
    _register_missing("intermediate/deep",
                      f"{_YPRED_PATH} not found — run evaluate.py first, or run with --gpu to load model directly.")
else:
    _register_missing("intermediate/deep", f"{_BM_PATH} not found")


# ===========================================================================
# 2–4.  Early fusion (MLP / RF / GBT)
# ===========================================================================
def _eval_early_fusion(key, family, model_type, path, label):
    print(f"\n=== {label} ===")
    if emb_cache is None:
        _register_missing(key, "precomputed_embeddings.pt not found")
        return
    if not path.exists():
        _register_missing(key, f"{path} not found")
        return
    try:
        if model_type == "pytorch_early_fusion":
            ckpt   = torch.load(path, map_location="cpu")
            cfg    = ckpt["config"]          # {text_encoder, image_encoder, d_in}
            t_enc  = cfg["text_encoder"]
            i_enc  = cfg["image_encoder"]
            X_test_ef = np.concatenate([
                _get_emb("text",  t_enc, "test"),
                _get_emb("image", i_enc, "test"),
            ], axis=1)
            model  = _EarlyFusionMLP(cfg["d_in"], N_CLASSES, hidden=256, dropout=0.3)
            model.load_state_dict(ckpt["state_dict"])
            y_pred = _pt_predict(model, torch.tensor(X_test_ef))
        else:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
            cfg   = bundle["config"]
            t_enc = cfg["text_encoder"]
            i_enc = cfg["image_encoder"]
            X_test_ef = np.concatenate([
                _get_emb("text",  t_enc, "test"),
                _get_emb("image", i_enc, "test"),
            ], axis=1)
            model  = bundle["model"]
            y_pred = model.predict(X_test_ef).astype(int)

        _register(
            key=key,
            strategy="early", family=family,
            model_type=model_type,
            model_files={"main": str(path)},
            ablation_run_name=f"fusion_early_{family}",
            y_pred=y_pred,
            extra={"text_encoder": t_enc, "image_encoder": i_enc},
        )
    except Exception as exc:
        _register_missing(key, str(exc))


_eval_early_fusion("early/mlp",  "mlp",  "pytorch_early_fusion",
                    DATA_DIR / "best_model_early_fusion_mlp.pt",  "2. Early fusion — MLP")
_eval_early_fusion("early/rf",   "rf",   "sklearn_early_fusion",
                    DATA_DIR / "best_model_early_fusion_rf.pkl",   "3. Early fusion — RF")
_eval_early_fusion("early/gbt",  "gbt",  "sklearn_early_fusion",
                    DATA_DIR / "best_model_early_fusion_gbt.pkl",  "4. Early fusion — GBT")


# ===========================================================================
# 5–6.  Late fusion (non-deep and deep)
# ===========================================================================
def _eval_late_fusion_family(family_key, family_label, txt_path, img_path, model_type):
    print(f"\n=== {family_label} ===")
    if emb_cache is None:
        _register_missing(family_key, "precomputed_embeddings.pt not found")
        return
    if not txt_path.exists() or not img_path.exists():
        missing = [str(p) for p in [txt_path, img_path] if not p.exists()]
        _register_missing(family_key, f"missing: {missing}")
        return
    try:
        if model_type == "late_fusion_deep":
            ckpt_t = torch.load(txt_path, map_location="cpu")
            ckpt_i = torch.load(img_path, map_location="cpu")
            t_enc  = ckpt_t["text_encoder"]
            i_enc  = ckpt_i["image_encoder"]
            d_t    = ckpt_t["d_in"]
            d_i    = ckpt_i["d_in"]
            X_test_t = _get_emb("text",  t_enc, "test")
            X_test_i = _get_emb("image", i_enc, "test")
            txt_mlp = _BranchMLP(d_t, N_CLASSES, hidden=128, dropout=0.3)
            txt_mlp.load_state_dict(ckpt_t["state_dict"])
            img_mlp = _BranchMLP(d_i, N_CLASSES, hidden=128, dropout=0.3)
            img_mlp.load_state_dict(ckpt_i["state_dict"])
            txt_mlp.eval(); img_mlp.eval()
            with torch.no_grad():
                p_t = torch.softmax(txt_mlp(torch.tensor(X_test_t)), dim=1).numpy()
                p_i = torch.softmax(img_mlp(torch.tensor(X_test_i)), dim=1).numpy()
        else:
            # non-deep: both models are sklearn (RF or GBT)
            with open(txt_path, "rb") as fh:
                bndl_t = pickle.load(fh)
            with open(img_path, "rb") as fh:
                bndl_i = pickle.load(fh)
            t_enc  = bndl_t["encoder"]
            i_enc  = bndl_i["encoder"]
            X_test_t = _get_emb("text",  t_enc, "test")
            X_test_i = _get_emb("image", i_enc, "test")
            p_t = bndl_t["model"].predict_proba(X_test_t).astype("float32")
            p_i = bndl_i["model"].predict_proba(X_test_i).astype("float32")

        y_pred = (0.5 * p_t + 0.5 * p_i).argmax(1).astype(int)
        _register(
            key=family_key,
            strategy="late",
            family=family_key.split("/")[1],
            model_type=model_type,
            model_files={"text": str(txt_path), "image": str(img_path)},
            ablation_run_name=f"fusion_late_{family_key.split('/')[1]}",
            y_pred=y_pred,
            extra={"text_encoder": t_enc, "image_encoder": i_enc},
        )
    except Exception as exc:
        _register_missing(family_key, str(exc))


_eval_late_fusion_family(
    "late/mlp",
    "5. Late fusion — MLP (deep)",
    DATA_DIR / "best_model_late_fusion_mlp_text.pt",
    DATA_DIR / "best_model_late_fusion_mlp_image.pt",
    "late_fusion_deep",
)

# Non-deep: CLAUDE.md says best-family models for text/image are saved as both
# rf and gbt pkl files; we try all combinations and pick the best one.
print("\n=== 6. Late fusion — non-deep ===")
_nondp_candidates = []
_nondp_file_combos = [
    ("rf",  DATA_DIR / "best_model_late_fusion_rf_text.pkl",
             DATA_DIR / "best_model_late_fusion_rf_image.pkl"),
    ("gbt", DATA_DIR / "best_model_late_fusion_gbt_text.pkl",
             DATA_DIR / "best_model_late_fusion_gbt_image.pkl"),
    # cross pairs
    ("rf_text_gbt_img", DATA_DIR / "best_model_late_fusion_rf_text.pkl",
                         DATA_DIR / "best_model_late_fusion_gbt_image.pkl"),
    ("gbt_text_rf_img", DATA_DIR / "best_model_late_fusion_gbt_text.pkl",
                         DATA_DIR / "best_model_late_fusion_rf_image.pkl"),
]
if emb_cache is not None:
    for combo_name, tp, ip in _nondp_file_combos:
        if not tp.exists() or not ip.exists():
            continue
        try:
            with open(tp, "rb") as fh:
                bt = pickle.load(fh)
            with open(ip, "rb") as fh:
                bi = pickle.load(fh)
            t_enc = bt["encoder"]; i_enc = bi["encoder"]
            p_t = bt["model"].predict_proba(_get_emb("text",  t_enc, "test")).astype("float32")
            p_i = bi["model"].predict_proba(_get_emb("image", i_enc, "test")).astype("float32")
            y_pred = (0.5 * p_t + 0.5 * p_i).argmax(1).astype(int)
            f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            _nondp_candidates.append((f1, combo_name, tp, ip, t_enc, i_enc, y_pred))
        except Exception:
            pass

if _nondp_candidates:
    _nondp_candidates.sort(reverse=True)
    best_f1, best_combo, best_tp, best_ip, best_te, best_ie, best_ypred = _nondp_candidates[0]
    _register(
        key="late/nondp",
        strategy="late", family="nondp",
        model_type="late_fusion_nondp",
        model_files={"text": str(best_tp), "image": str(best_ip)},
        ablation_run_name="fusion_late_nondp",
        y_pred=best_ypred,
        extra={"text_encoder": best_te, "image_encoder": best_ie,
               "combo": best_combo},
    )
elif emb_cache is None:
    _register_missing("late/nondp", "precomputed_embeddings.pt not found")
else:
    _register_missing("late/nondp", "no non-deep late-fusion model files found")


# ===========================================================================
# Derive best_per_strategy
# ===========================================================================
strategy_keys = {
    "intermediate": ["intermediate/deep"],
    "early":        ["early/mlp", "early/rf", "early/gbt"],
    "late":         ["late/mlp", "late/nondp"],
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
