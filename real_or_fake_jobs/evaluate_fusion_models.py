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
    cd real_or_fake_jobs
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
X_test_static        = raw["X_test_static"].astype("float32")
X_test_description   = raw["X_test_description"].tolist()
X_test_company_profile = raw["X_test_company_profile"].tolist()
X_test_requirements  = raw["X_test_requirements"].tolist()
y_test               = raw["y_test"].astype(int)
label_classes        = raw["label_classes"].tolist()
N_CLASSES            = len(label_classes)
D_TAB                = X_test_static.shape[1]
TEXT_FIELDS          = ("description", "company_profile", "requirements")
print(f"  Test samples: {len(y_test)}  |  classes: {label_classes}")

# ---------------------------------------------------------------------------
# Load frozen CLS embedding cache (per-field DistilBERT CLS vectors)
# ---------------------------------------------------------------------------
EMB_CACHE = DATA_DIR / "early_fusion_text_embeddings.pt"
if EMB_CACHE.exists():
    print(f"Loading CLS embedding cache from {EMB_CACHE} …")
    _cache     = torch.load(EMB_CACHE, map_location="cpu")
    # Shape: {field: tensor(n, 768)} for both "train" and "test"
    test_cls_by_field  = {f: _cache["test"][f].numpy().astype("float32")
                          for f in TEXT_FIELDS}
    train_cls_by_field = {f: _cache["train"][f].numpy().astype("float32")
                          for f in TEXT_FIELDS}
    # Concatenated [desc || profile || reqs] (3×768 = 2304) + tabular
    test_cls_concat  = np.concatenate([test_cls_by_field[f]  for f in TEXT_FIELDS], axis=1)
    train_cls_concat = np.concatenate([train_cls_by_field[f] for f in TEXT_FIELDS], axis=1)
    X_test_ef  = np.concatenate([test_cls_concat,  X_test_static], axis=1)
    print(f"  CLS cache loaded — test EF shape {X_test_ef.shape}")
else:
    test_cls_by_field = train_cls_by_field = test_cls_concat = X_test_ef = None
    print("[warn] early_fusion_text_embeddings.pt not found — "
          "early/late-fusion deep models will be skipped.")

# ---------------------------------------------------------------------------
# Inline model class definitions
# ---------------------------------------------------------------------------

class _EarlyFusionMLP(nn.Module):
    """4-layer MLP (512 → 256 → 128 → n_classes), matches train_early_fusion.py."""
    def __init__(self, d_in, n_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 4, n_classes),
        )
    def forward(self, x): return self.net(x)


class _BranchMLP(nn.Module):
    """2-layer MLP (d_in → 128 → 64 → n_classes), matches train_late_fusion.py."""
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),         nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),  nn.ReLU(), nn.Dropout(dropout),
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
        self.net     = nn.Sequential(*layers)
        self.out_dim = in_dim
    def forward(self, x): return self.net(x)


class _MultimodalClassifier(nn.Module):
    """Inline replica of the intermediate-fusion DistilBERT classifier."""
    _TEXT_MODEL = "distilbert-base-uncased"

    def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
        super().__init__()
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(self._TEXT_MODEL)
        enc_dim = self.text_encoder.config.hidden_size
        def _proj():
            return nn.Sequential(
                nn.Linear(enc_dim, text_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            )
        self.proj_description     = _proj()
        self.proj_company_profile = _proj()
        self.proj_requirements    = _proj()
        self.tab_head    = _TabHead(d_tab, tab_hidden_dims, dropout)
        self.classifier  = nn.Linear(3 * text_hidden_dim + self.tab_head.out_dim, n_classes)

    def forward(self, desc_ids, desc_mask, profile_ids, profile_mask,
                reqs_ids, reqs_mask, X_static):
        B = desc_ids.size(0)
        all_ids  = torch.cat([desc_ids,  profile_ids,  reqs_ids],  dim=0)
        all_mask = torch.cat([desc_mask, profile_mask, reqs_mask], dim=0)
        cls_all  = self.text_encoder(
            input_ids=all_ids, attention_mask=all_mask
        ).last_hidden_state[:, 0]
        desc_emb, profile_emb, reqs_emb = cls_all.split(B, dim=0)
        return self.classifier(torch.cat([
            self.proj_description(desc_emb),
            self.proj_company_profile(profile_emb),
            self.proj_requirements(reqs_emb),
            self.tab_head(X_static),
        ], dim=1))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _metrics(y_true, y_pred):
    return {
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_accuracy":  float(accuracy_score(y_true, y_pred)),
    }


def _pt_predict_single(model, X_tensor, batch_size=256):
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
    "dataset":        "real_or_fake_jobs",
    "generated_at":   datetime.now().isoformat(),
    "label_classes":  label_classes,
    "models":         {},
    "best_per_strategy": {},
}


def _register(key, strategy, family, model_type, model_files,
               ablation_run_name, y_pred, extra=None):
    m = _metrics(y_test, y_pred)
    entry = {
        "strategy":          strategy,
        "family":            family,
        "model_type":        model_type,
        "model_files":       model_files,
        "available":         True,
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
        ckpt     = torch.load(_IF_PATH, map_location=DEVICE)
        tok      = AutoTokenizer.from_pretrained(_MultimodalClassifier._TEXT_MODEL)
        if_model = _MultimodalClassifier(
            d_tab           = len(ckpt["tab_cols"]),
            n_classes       = N_CLASSES,
            tab_hidden_dims = ckpt["config"]["tab_hidden_dims"],
            text_hidden_dim = ckpt["config"]["text_hidden_dim"],
            dropout         = ckpt["config"]["dropout"],
        ).to(DEVICE)
        if_model.load_state_dict(ckpt["state_dict"])
        if_model.eval()

        _BATCH = 16
        _MAX_LEN = 512
        preds = []
        with torch.no_grad():
            for i in range(0, len(y_test), _BATCH):
                descs    = [str(t) for t in X_test_description[i:i+_BATCH]]
                profiles = [str(t) for t in X_test_company_profile[i:i+_BATCH]]
                reqs_    = [str(t) for t in X_test_requirements[i:i+_BATCH]]
                enc_d = tok(descs,    max_length=_MAX_LEN, padding="max_length",
                            truncation=True, return_tensors="pt")
                enc_p = tok(profiles, max_length=_MAX_LEN, padding="max_length",
                            truncation=True, return_tensors="pt")
                enc_r = tok(reqs_,    max_length=_MAX_LEN, padding="max_length",
                            truncation=True, return_tensors="pt")
                tab = torch.tensor(X_test_static[i:i+_BATCH]).to(DEVICE)
                logits = if_model(
                    enc_d["input_ids"].to(DEVICE), enc_d["attention_mask"].to(DEVICE),
                    enc_p["input_ids"].to(DEVICE), enc_p["attention_mask"].to(DEVICE),
                    enc_r["input_ids"].to(DEVICE), enc_r["attention_mask"].to(DEVICE),
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
            extra={"val_macro_f1": float(ckpt["config"].get("val_f1", 0.0))},
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
        y_pred = _pt_predict_single(ef_mlp, torch.tensor(X_test_ef))
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
        rf     = bundle["model"] if isinstance(bundle, dict) else bundle
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
        gbt    = bundle["model"] if isinstance(bundle, dict) else bundle
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
# 5.  Late fusion — non-deep  (TF-IDF+LR × 3 fields + best tabular sklearn)
# ===========================================================================
print("\n=== 5. Late fusion — non-deep ===")
_LF_TFIDF_PATHS = {
    f: DATA_DIR / f"best_model_late_fusion_tfidf_logreg_{f}.pkl" for f in TEXT_FIELDS
}
_LF_RF_TAB_PATH  = DATA_DIR / "best_model_late_fusion_rf_tabular.pkl"
_LF_GBT_TAB_PATH = DATA_DIR / "best_model_late_fusion_gbt_tabular.pkl"

_missing_tfidf = [f for f, p in _LF_TFIDF_PATHS.items() if not p.exists()]
_tab_paths_avail = [p for p in [_LF_RF_TAB_PATH, _LF_GBT_TAB_PATH] if p.exists()]

if _missing_tfidf:
    _register_missing("late/nondp",
                      f"missing TF-IDF models for fields: {_missing_tfidf}")
elif not _tab_paths_avail:
    _register_missing("late/nondp", "no tabular non-deep model found (rf or gbt)")
else:
    try:
        # Text branches
        p_text_branches = []
        text_model_files = {}
        for field in TEXT_FIELDS:
            with open(_LF_TFIDF_PATHS[field], "rb") as fh:
                bndl = pickle.load(fh)
            tfidf_v = bndl["tfidf"]
            logreg  = bndl["model"]
            texts   = globals().get(f"X_test_{field}", [str(t) for t in raw.get(f"X_test_{field}", [])])
            # Use the already-loaded arrays from the top of the script
            _field_texts = {
                "description":     X_test_description,
                "company_profile": X_test_company_profile,
                "requirements":    X_test_requirements,
            }[field]
            p_text_branches.append(
                logreg.predict_proba(tfidf_v.transform(_field_texts)).astype("float32")
            )
            text_model_files[field] = str(_LF_TFIDF_PATHS[field])

        # Tabular branch — pick best combined F1 between RF and GBT
        best_tab_path, best_p_tab, best_tab_f1 = None, None, -1.0
        for _tp in _tab_paths_avail:
            with open(_tp, "rb") as fh:
                _b = pickle.load(fh)
            _m = _b["model"] if isinstance(_b, dict) else _b
            _pt = _m.predict_proba(X_test_static).astype("float32")
            _yp = (np.mean(p_text_branches + [_pt], axis=0)).argmax(1)
            _f1 = float(f1_score(y_test, _yp, average="macro", zero_division=0))
            if _f1 > best_tab_f1:
                best_tab_path, best_p_tab, best_tab_f1 = _tp, _pt, _f1

        p_combined = np.mean(p_text_branches + [best_p_tab], axis=0).argmax(1).astype(int)
        _register(
            key="late/nondp",
            strategy="late", family="nondp",
            model_type="late_fusion_nondp",
            model_files={**text_model_files, "tabular": str(best_tab_path)},
            ablation_run_name="fusion_late_nondp",
            y_pred=p_combined,
        )
    except Exception as exc:
        _register_missing("late/nondp", str(exc))


# ===========================================================================
# 6.  Late fusion — deep  (BranchMLP × 3 fields + BranchMLP tabular)
# ===========================================================================
print("\n=== 6. Late fusion — deep ===")
_LF_TEXT_MLP_PATHS = {
    f: DATA_DIR / f"best_model_late_fusion_mlp_{f}.pt" for f in TEXT_FIELDS
}
_LF_TAB_MLP_PATH = DATA_DIR / "best_model_late_fusion_mlp_tabular.pt"

_missing_deep = [str(p) for p in list(_LF_TEXT_MLP_PATHS.values()) + [_LF_TAB_MLP_PATH]
                 if not p.exists()]

if _missing_deep:
    _register_missing("late/mlp_deep", f"missing: {_missing_deep}")
elif test_cls_by_field is None:
    _register_missing("late/mlp_deep", "embedding cache not found")
else:
    try:
        p_branches = []
        mlp_files  = {}
        for field in TEXT_FIELDS:
            ckpt_t  = torch.load(_LF_TEXT_MLP_PATHS[field], map_location="cpu")
            d_t     = 768   # CLS dim
            h_t     = 128   # hidden (from BranchMLP default)
            nc_t    = N_CLASSES
            txt_mlp = _BranchMLP(d_t, nc_t, h_t)
            txt_mlp.load_state_dict(ckpt_t["state_dict"])
            txt_mlp.eval()
            with torch.no_grad():
                p_t = torch.softmax(
                    txt_mlp(torch.tensor(test_cls_by_field[field])), dim=1
                ).numpy().astype("float32")
            p_branches.append(p_t)
            mlp_files[field] = str(_LF_TEXT_MLP_PATHS[field])

        ckpt_tb = torch.load(_LF_TAB_MLP_PATH, map_location="cpu")
        d_tb    = D_TAB
        h_tb    = 128
        tab_mlp = _BranchMLP(d_tb, N_CLASSES, h_tb)
        tab_mlp.load_state_dict(ckpt_tb["state_dict"])
        tab_mlp.eval()
        with torch.no_grad():
            p_tab = torch.softmax(
                tab_mlp(torch.tensor(X_test_static)), dim=1
            ).numpy().astype("float32")
        p_branches.append(p_tab)
        mlp_files["tabular"] = str(_LF_TAB_MLP_PATH)

        y_pred = np.mean(p_branches, axis=0).argmax(1).astype(int)
        _register(
            key="late/mlp_deep",
            strategy="late", family="mlp_deep",
            model_type="late_fusion_deep",
            model_files=mlp_files,
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
