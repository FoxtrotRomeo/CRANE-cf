"""
Check whether a distance ablation has been run for the best model in each
fusion strategy, and launch missing ablations.

Reads: data/fusion_model_registry.json  (written by evaluate_fusion_models.py)
Writes: data/ablation_runs/fusion_{strategy}_{family}/summary.json per strategy

For each fusion strategy:
  1. Read the registry to find the best model (highest test macro-F1).
  2. Check whether data/ablation_runs/<ablation_run_name>/summary.json exists.
  3. If absent, build the appropriate predict_fn for that model and run the
     full distance ablation (same encoders, metrics and generator set as
     run_cf_ablation.py, minus IntermediateFusion for non-deep models).

Run
---
    cd real_or_fake_jobs
    python run_cf_for_best_models.py --gpu 7
    python run_cf_for_best_models.py --gpu 7 --no-bert
    python run_cf_for_best_models.py --gpu 7 --dry-run   # check only
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from job_cf_factory import build_job_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import CombinedNN, EarlyFusionNN
from cf_lib.unimodal import TabularNN, TextNN
from counterfactual_helpers import find_k_closest_latent, find_k_closest_static
from sklearn.metrics import pairwise_distances as _pairwise_dist

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run missing CF ablations for the best model in each fusion strategy."
)
parser.add_argument("--gpu",              type=int,   default=7)
parser.add_argument("--k",               type=int,   default=20)
parser.add_argument("--max-samples",     type=int,   default=None)
parser.add_argument("--n-jobs",          type=int,   default=1)
parser.add_argument("--output-dir",      type=str,   default="data/ablation_runs")
parser.add_argument("--no-bert",         action="store_true")
parser.add_argument("--word2vec-path",   type=str,   default="data/word2vec_google_news_300.kv")
parser.add_argument("--source-class",   type=str,   default="fake")
parser.add_argument("--target-class",   type=str,   default="real")
parser.add_argument("--save-full",       action="store_true")
parser.add_argument("--dry-run",         action="store_true",
                    help="Print which ablations would be launched without running them.")
parser.add_argument("--strategies",      type=str,   default=None,
                    help="Comma-separated subset of strategies to process.")
args = parser.parse_args()

DATA_DIR = Path("data")
DEVICE = (
    f"cuda:{args.gpu}"
    if args.gpu is not None and torch.cuda.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

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

label_classes = registry["label_classes"]
lc_lower      = [c.lower() for c in label_classes]
source_value  = lc_lower.index(args.source_class.lower())
target_value  = lc_lower.index(args.target_class.lower())

# ---------------------------------------------------------------------------
# Resolve strategies to process
# ---------------------------------------------------------------------------
all_strategies = ["intermediate", "early", "late"]
if args.strategies:
    all_strategies = [s.strip() for s in args.strategies.split(",")]

todo = []
for strategy in all_strategies:
    best_key = registry["best_per_strategy"].get(strategy)
    if best_key is None:
        print(f"[{strategy}] No available model in registry — skipping.")
        continue
    entry = registry["models"][best_key]
    run_name = entry["ablation_run_name"] + "_k50"
    summary_path = Path(args.output_dir) / run_name / "summary.json"
    if summary_path.exists():
        print(f"[{strategy}] Ablation already exists: {summary_path} — skipping.")
        continue
    todo.append((strategy, best_key, entry, run_name))
    print(f"[{strategy}] Will run ablation '{run_name}'  "
          f"(model_type={entry['model_type']}  F1={entry.get('test_macro_f1', 'N/A'):.4f})")

if not todo:
    print("\nAll ablations are up to date.")
    sys.exit(0)

if args.dry_run:
    print("\nDry run — exiting without launching ablations.")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Shared setup: load dataset, text backends, embeddings, LOF
# ---------------------------------------------------------------------------
print("\nLoading dataset and text backends …")
produced = build_job_dataset(gpu=args.gpu, load_bert=not args.no_bert)
dataset       = produced["dataset"]
text_bk       = produced["text_backend_kwargs"]
y_pred_orig   = produced["y_pred"]

X_train_description     = produced["X_train_description"]
X_test_description      = produced["X_test_description"]
X_train_company_profile = produced["X_train_company_profile"]
X_test_company_profile  = produced["X_test_company_profile"]
X_train_requirements    = produced["X_train_requirements"]
X_test_requirements     = produced["X_test_requirements"]

_FIELD_NAMES  = ["description", "company_profile", "requirements"]
_FIELD_TRAINS = [X_train_description, X_train_company_profile, X_train_requirements]
_FIELD_TESTS  = [X_test_description,  X_test_company_profile,  X_test_requirements]

# Source indices (samples to explain)
if y_pred_orig is not None:
    source_indices = [int(i) for i, p in enumerate(y_pred_orig) if int(p) == source_value]
else:
    source_indices = [int(i) for i in range(len(dataset.y_test))
                      if int(dataset.y_test[i]) == source_value]
if args.max_samples is not None:
    source_indices = source_indices[:args.max_samples]
print(f"Source indices: {len(source_indices)} samples predicted as '{args.source_class}'")

# ---------------------------------------------------------------------------
# TF-IDF (shared across all combos)
# ---------------------------------------------------------------------------
print("Fitting TF-IDF on all three text fields …")
_all_train_texts = (
    [str(t) for t in X_train_description]
    + [str(t) for t in X_train_company_profile]
    + [str(t) for t in X_train_requirements]
)
_tfidf_vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
_tfidf_vec.fit(_all_train_texts)

def _tfidf_embed_fn(texts):
    return _tfidf_vec.transform(texts).toarray().astype(np.float32)

# ---------------------------------------------------------------------------
# Tabular LOF (pre-fitted once)
# ---------------------------------------------------------------------------
from sklearn.neighbors import LocalOutlierFactor as _LOF
print("Pre-fitting tabular LOF …")
_n_lof = max(2, min(20, len(dataset.X_train_static) - 1))
_tab_lof = _LOF(n_neighbors=_n_lof, novelty=True)
_tab_lof.fit(dataset.X_train_static)
_tab_lof_train_scores = -_tab_lof.score_samples(dataset.X_train_static)
_tab_lof_low  = float(np.percentile(_tab_lof_train_scores, 5))
_tab_lof_high = float(np.percentile(_tab_lof_train_scores, 95))

# ---------------------------------------------------------------------------
# BERT embed_fn
# ---------------------------------------------------------------------------
_bert_embed_fn: Optional[object] = None
if not args.no_bert:
    from counterfactual_evaluation_helpers import _make_embed_fn_from_e5_kwargs
    _bert_embed_fn = _make_embed_fn_from_e5_kwargs(
        tokenizer=text_bk["bert_tokenizer"],
        model=text_bk["bert_model"],
        device=text_bk["bert_device"],
    )

# word2vec
_w2v_embed_fn: Optional[object] = None
_w2v_kv = None
if Path(args.word2vec_path).exists():
    from gensim.models import KeyedVectors
    print(f"Loading word2vec from {args.word2vec_path} …")
    _w2v_kv = (
        KeyedVectors.load(args.word2vec_path)
        if args.word2vec_path.endswith(".kv")
        else KeyedVectors.load_word2vec_format(
            args.word2vec_path, binary=args.word2vec_path.endswith(".bin")
        )
    )
    def _w2v_embed_fn(texts, _kv=_w2v_kv):
        out = []
        for text in texts:
            tokens = str(text).lower().split()
            vecs   = [_kv[t] for t in tokens if t in _kv]
            emb    = (np.mean(vecs, axis=0).astype(np.float32) if vecs
                      else np.zeros(_kv.vector_size, dtype=np.float32))
            out.append(emb)
        return np.stack(out)
    text_bk["word2vec_model"] = _w2v_kv
    print(f"  word2vec loaded: {_w2v_kv.vector_size}-dim")

# Text encoders list
text_encoders = ["tfidf", "raw"]
if _bert_embed_fn is not None:
    text_encoders.insert(0, "bert")
if _w2v_embed_fn is not None:
    text_encoders.insert(-1, "word2vec")
tab_metrics = ["euclidean", "manhattan"]

_encoder_embed_fns = [("tfidf", _tfidf_embed_fn)]
if _bert_embed_fn is not None:
    _encoder_embed_fns.append(("bert", _bert_embed_fn))
if _w2v_embed_fn is not None:
    _encoder_embed_fns.append(("word2vec", _w2v_embed_fn))

# ---------------------------------------------------------------------------
# Precompute per-field embeddings and concat for EarlyFusion
# ---------------------------------------------------------------------------
def _precompute_field(name, embed_fn, train_arr, test_arr):
    all_texts = [str(t) for t in train_arr] + [str(t) for t in test_arr]
    print(f"  Precomputing {name} for {len(all_texts)} texts …")
    all_embs = np.asarray(embed_fn(all_texts), dtype=np.float32)
    n_train  = len(train_arr)
    return {"train": all_embs[:n_train].copy(), "test": all_embs[n_train:].copy()}

_precomputed_by_field: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
    fname: {} for fname in _FIELD_NAMES
}
for enc_name, embed_fn in _encoder_embed_fns:
    for fname, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS):
        _precomputed_by_field[fname][enc_name] = _precompute_field(
            f"{fname}/{enc_name}", embed_fn, ftrain, ftest
        )

_precomputed_concat: Dict[str, Dict[str, np.ndarray]] = {}
for enc_name, _ in _encoder_embed_fns:
    _precomputed_concat[enc_name] = {
        split: np.concatenate(
            [_precomputed_by_field[fname][enc_name][split] for fname in _FIELD_NAMES], axis=1
        )
        for split in ("train", "test")
    }

# Per-field text→vector lookups (for objective evaluation)
_field_emb_lookup: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
    fname: {
        enc_name: {
            **{str(t): v for t, v in zip(ftrain, _precomputed_by_field[fname][enc_name]["train"])},
            **{str(t): v for t, v in zip(ftest,  _precomputed_by_field[fname][enc_name]["test"])},
        }
        for enc_name, _ in _encoder_embed_fns
    }
    for fname, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS)
}

text_bk["precomputed_text_embeddings_by_encoder"] = {
    fname: {enc: _precomputed_by_field[fname][enc] for enc, _ in _encoder_embed_fns}
    for fname in _FIELD_NAMES
}
text_bk["auto_text_branch_generators"]  = False
text_bk["precompute_all_text_branches"] = True

# ---------------------------------------------------------------------------
# Load CLS embedding cache (for early/late deep fusion predict_fn)
# ---------------------------------------------------------------------------
_EMB_CACHE = DATA_DIR / "early_fusion_text_embeddings.pt"
_train_cls_by_field: Optional[Dict[str, np.ndarray]] = None
_test_cls_by_field:  Optional[Dict[str, np.ndarray]] = None
if _EMB_CACHE.exists():
    print(f"Loading CLS cache from {_EMB_CACHE} …")
    _ec = torch.load(_EMB_CACHE, map_location="cpu")
    _train_cls_by_field = {f: _ec["train"][f].numpy().astype("float32")
                           for f in _FIELD_NAMES}
    _test_cls_by_field  = {f: _ec["test"][f].numpy().astype("float32")
                           for f in _FIELD_NAMES}
    # Build text→CLS lookup per field
    _cls_lookup: Dict[str, Dict[str, np.ndarray]] = {
        field: {
            **{str(t): v for t, v in zip(ftrain, _train_cls_by_field[field])},
            **{str(t): v for t, v in zip(ftest,  _test_cls_by_field[field])},
        }
        for field, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS)
    }
    print("  CLS cache loaded.")
else:
    _cls_lookup = {}
    print("[warn] CLS cache not found — early/late deep fusion predict_fns will use live BERT.")

# Description → (desc, profile, reqs) lookup for single-string fallback
_desc_to_fields: dict = {}
for _d, _p, _r in zip(X_train_description, X_train_company_profile, X_train_requirements):
    _desc_to_fields.setdefault(str(_d), (str(_d), str(_p), str(_r)))

# ===========================================================================
# Predict-fn builders (one per model_type)
# ===========================================================================

def _resolve_texts(x_tab_or_dict, text_candidate_or_dict):
    """Return (desc, profile, reqs, x_tab) from the cf-lib call signature."""
    if isinstance(text_candidate_or_dict, dict):
        desc    = str(text_candidate_or_dict.get("description",    "") or "")
        profile = str(text_candidate_or_dict.get("company_profile","") or "")
        reqs    = str(text_candidate_or_dict.get("requirements",   "") or "")
        x_tab   = (x_tab_or_dict.get("__primary__")
                   if isinstance(x_tab_or_dict, dict) else x_tab_or_dict)
    else:
        text = str(text_candidate_or_dict) if text_candidate_or_dict is not None else ""
        desc, profile, reqs = _desc_to_fields.get(text, (text, "", ""))
        x_tab = x_tab_or_dict
    return desc, profile, reqs, x_tab


def _get_cls(field: str, text: str) -> np.ndarray:
    """Return CLS vector for a text string from the lookup dict."""
    lk = _cls_lookup.get(field, {})
    v  = lk.get(text)
    if v is None:
        # Live inference fallback (slow; only for Frankenstein hybrids)
        if _bert_embed_fn is not None:
            v = np.asarray(_bert_embed_fn([text]), dtype=np.float32)[0]
        else:
            v = np.zeros(768, dtype=np.float32)
    return v


def _build_predict_fn_intermediate(entry: dict):
    """predict_fn for pytorch_intermediate (DistilBERT multi-text classifier)."""
    model_path = Path(entry["model_files"]["main"])
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found")

    from transformers import AutoModel, AutoTokenizer

    _TEXT_MODEL = "distilbert-base-uncased"

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

    class _MC(nn.Module):
        def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
            super().__init__()
            self.text_encoder = AutoModel.from_pretrained(_TEXT_MODEL)
            enc_dim  = self.text_encoder.config.hidden_size
            def _p():
                return nn.Sequential(nn.Linear(enc_dim, text_hidden_dim),
                                     nn.ReLU(), nn.Dropout(dropout))
            self.proj_description     = _p()
            self.proj_company_profile = _p()
            self.proj_requirements    = _p()
            self.tab_head  = _TabHead(d_tab, tab_hidden_dims, dropout)
            self.classifier = nn.Linear(3*text_hidden_dim+self.tab_head.out_dim, n_classes)
        def forward(self, di, dm, pi, pm, ri, rm, tab):
            B = di.size(0)
            cls_all = self.text_encoder(
                input_ids=torch.cat([di, pi, ri], 0),
                attention_mask=torch.cat([dm, pm, rm], 0),
            ).last_hidden_state[:, 0]
            de, pe, re = cls_all.split(B, 0)
            return self.classifier(torch.cat([
                self.proj_description(de), self.proj_company_profile(pe),
                self.proj_requirements(re), self.tab_head(tab),
            ], 1))

    ckpt  = torch.load(model_path, map_location=DEVICE)
    model = _MC(
        d_tab           = len(ckpt["tab_cols"]),
        n_classes       = len(ckpt["label_classes"]),
        tab_hidden_dims = ckpt["config"]["tab_hidden_dims"],
        text_hidden_dim = ckpt["config"]["text_hidden_dim"],
        dropout         = ckpt["config"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tok = AutoTokenizer.from_pretrained(_TEXT_MODEL)
    _ML = 512
    lock = threading.Lock()

    # Build training-sample prediction cache
    print("  Building training-sample prediction cache …")
    X_tr_static = np.asarray(dataset.X_train_static, dtype=np.float32)
    pred_rows = []
    with torch.no_grad():
        for bs in range(0, len(X_tr_static), 16):
            be = min(bs + 16, len(X_tr_static))
            descs    = [str(t) for t in X_train_description[bs:be]]
            profiles = [str(t) for t in X_train_company_profile[bs:be]]
            reqs_    = [str(t) for t in X_train_requirements[bs:be]]
            ed = tok(descs,    max_length=_ML, padding="max_length", truncation=True, return_tensors="pt")
            ep = tok(profiles, max_length=_ML, padding="max_length", truncation=True, return_tensors="pt")
            er = tok(reqs_,    max_length=_ML, padding="max_length", truncation=True, return_tensors="pt")
            tab = torch.tensor(X_tr_static[bs:be]).to(DEVICE)
            logits = model(
                ed["input_ids"].to(DEVICE), ed["attention_mask"].to(DEVICE),
                ep["input_ids"].to(DEVICE), ep["attention_mask"].to(DEVICE),
                er["input_ids"].to(DEVICE), er["attention_mask"].to(DEVICE), tab,
            )
            pred_rows.append(logits.argmax(1).cpu().numpy().astype(np.float32))
    _train_preds = np.concatenate(pred_rows)
    _train_cache = {
        (X_tr_static[i].tobytes(), str(X_train_description[i]),
         str(X_train_company_profile[i]), str(X_train_requirements[i])): float(_train_preds[i])
        for i in range(len(X_tr_static))
    }
    print(f"  Cached {len(_train_cache):,} training-sample predictions.")

    # Latents for IntermediateFusion generator
    print("  Extracting model latents …")
    _buf: dict = {}
    def _hook(m, inp, out): _buf["z"] = inp[0].detach().cpu().numpy()
    handle = model.classifier.register_forward_hook(_hook)
    def _run_latents(X_static, X_desc, X_prof, X_reqs):
        rows = []
        with torch.no_grad():
            for bs in range(0, len(X_static), 16):
                be = min(bs + 16, len(X_static))
                ed = tok([str(t) for t in X_desc[bs:be]], max_length=_ML, padding="max_length",
                         truncation=True, return_tensors="pt")
                ep = tok([str(t) for t in X_prof[bs:be]], max_length=_ML, padding="max_length",
                         truncation=True, return_tensors="pt")
                er = tok([str(t) for t in X_reqs[bs:be]], max_length=_ML, padding="max_length",
                         truncation=True, return_tensors="pt")
                tab = torch.tensor(X_static[bs:be].astype(np.float32)).to(DEVICE)
                model(ed["input_ids"].to(DEVICE), ed["attention_mask"].to(DEVICE),
                      ep["input_ids"].to(DEVICE), ep["attention_mask"].to(DEVICE),
                      er["input_ids"].to(DEVICE), er["attention_mask"].to(DEVICE), tab)
                rows.append(_buf["z"].copy())
        return np.vstack(rows).astype(np.float32)
    train_latents = _run_latents(X_tr_static, X_train_description,
                                  X_train_company_profile, X_train_requirements)
    test_latents  = _run_latents(dataset.X_test_static, X_test_description,
                                  X_test_company_profile, X_test_requirements)
    handle.remove()

    def _predict_fn(x_tab_or_dict, x_ts, text_candidate_or_dict):
        desc, profile, reqs, x_tab = _resolve_texts(x_tab_or_dict, text_candidate_or_dict)
        if x_tab is not None:
            _key = (np.asarray(x_tab, dtype=np.float32).tobytes(), desc, profile, reqs)
            hit = _train_cache.get(_key)
            if hit is not None:
                return hit
        import torch as _t
        def _enc(t):
            return tok([t], max_length=_ML, padding="max_length",
                       truncation=True, return_tensors="pt")
        tab = _t.tensor(np.asarray(x_tab, dtype=np.float32)[None, :]).to(DEVICE)
        with lock:
            ed, ep, er = _enc(desc), _enc(profile), _enc(reqs)
            with _t.no_grad():
                logits = model(
                    ed["input_ids"].to(DEVICE), ed["attention_mask"].to(DEVICE),
                    ep["input_ids"].to(DEVICE), ep["attention_mask"].to(DEVICE),
                    er["input_ids"].to(DEVICE), er["attention_mask"].to(DEVICE), tab,
                )
        return float(logits.argmax(1).cpu().item())

    return _predict_fn, train_latents, test_latents


def _build_predict_fn_early(entry: dict):
    """predict_fn for pytorch_early_fusion and sklearn_early_fusion."""
    model_path  = Path(entry["model_files"]["main"])
    model_type  = entry["model_type"]
    D_TAB       = dataset.X_train_static.shape[1]
    X_tr_static = np.asarray(dataset.X_train_static, dtype=np.float32)

    if model_type == "pytorch_early_fusion":
        ckpt = torch.load(model_path, map_location="cpu")
        class _EarlyFusionMLP(nn.Module):
            def __init__(self, d_in, n_classes, hidden=512, dropout=0.3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden//4, n_classes),
                )
            def forward(self, x): return self.net(x)
        model = _EarlyFusionMLP(ckpt["d_in"], ckpt["n_classes"], ckpt["hidden"], ckpt["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        def _run(X_ef):
            with torch.no_grad():
                return int(model(torch.tensor(X_ef, dtype=torch.float32)).argmax(1).item())
    else:
        with open(model_path, "rb") as fh:
            bundle = pickle.load(fh)
        model = bundle["model"] if isinstance(bundle, dict) else bundle
        def _run(X_ef):
            return int(model.predict(X_ef)[0])

    def _predict_fn(x_tab_or_dict, x_ts, text_candidate_or_dict):
        desc, profile, reqs, x_tab = _resolve_texts(x_tab_or_dict, text_candidate_or_dict)
        cls_desc    = _get_cls("description",     desc)
        cls_profile = _get_cls("company_profile", profile)
        cls_reqs    = _get_cls("requirements",    reqs)
        x_tab_arr   = np.asarray(x_tab if x_tab is not None else
                                 np.zeros(D_TAB, dtype=np.float32), dtype=np.float32)
        ef_vec = np.concatenate([cls_desc, cls_profile, cls_reqs, x_tab_arr]).reshape(1, -1)
        return float(_run(ef_vec))

    return _predict_fn, None, None


def _build_predict_fn_late_nondp(entry: dict):
    """predict_fn for late_fusion_nondp (TF-IDF+LR × 3 + tabular sklearn)."""
    model_files = entry["model_files"]
    # Load text branches
    text_models = {}
    for field in _FIELD_NAMES:
        with open(Path(model_files[field]), "rb") as fh:
            bndl = pickle.load(fh)
        text_models[field] = (bndl["tfidf"], bndl["model"])
    # Load tabular branch
    with open(Path(model_files["tabular"]), "rb") as fh:
        tb = pickle.load(fh)
    tab_model = tb["model"] if isinstance(tb, dict) else tb
    D_TAB = dataset.X_train_static.shape[1]

    def _predict_fn(x_tab_or_dict, x_ts, text_candidate_or_dict):
        desc, profile, reqs, x_tab = _resolve_texts(x_tab_or_dict, text_candidate_or_dict)
        texts = {"description": desc, "company_profile": profile, "requirements": reqs}
        p_branches = []
        for field in _FIELD_NAMES:
            tfidf, logreg = text_models[field]
            p_branches.append(logreg.predict_proba(tfidf.transform([texts[field]]))[0])
        x_tab_arr = np.asarray(x_tab if x_tab is not None else
                               np.zeros(D_TAB, dtype=np.float32),
                               dtype=np.float32).reshape(1, -1)
        p_branches.append(tab_model.predict_proba(x_tab_arr)[0])
        return float(np.mean(p_branches, axis=0).argmax())

    return _predict_fn, None, None


def _build_predict_fn_late_deep(entry: dict):
    """predict_fn for late_fusion_deep (BranchMLP × 3 text + BranchMLP tabular)."""
    model_files = entry["model_files"]
    D_TAB = dataset.X_train_static.shape[1]
    N_CLS = len(label_classes)

    class _BranchMLP(nn.Module):
        def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden//2, n_classes),
            )
        def forward(self, x): return self.net(x)

    text_mlps = {}
    for field in _FIELD_NAMES:
        ckpt = torch.load(Path(model_files[field]), map_location="cpu")
        m = _BranchMLP(768, N_CLS, 128)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        text_mlps[field] = m

    ckpt_tab = torch.load(Path(model_files["tabular"]), map_location="cpu")
    tab_mlp  = _BranchMLP(D_TAB, N_CLS, 128)
    tab_mlp.load_state_dict(ckpt_tab["state_dict"])
    tab_mlp.eval()

    def _predict_fn(x_tab_or_dict, x_ts, text_candidate_or_dict):
        desc, profile, reqs, x_tab = _resolve_texts(x_tab_or_dict, text_candidate_or_dict)
        texts = {"description": desc, "company_profile": profile, "requirements": reqs}
        p_branches = []
        for field in _FIELD_NAMES:
            cls_vec = _get_cls(field, texts[field])
            with torch.no_grad():
                p_t = torch.softmax(
                    text_mlps[field](torch.tensor(cls_vec).unsqueeze(0)), dim=1
                ).numpy()[0]
            p_branches.append(p_t)
        x_tab_arr = np.asarray(x_tab if x_tab is not None else
                               np.zeros(D_TAB, dtype=np.float32),
                               dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            p_tab = torch.softmax(
                tab_mlp(torch.tensor(x_tab_arr, dtype=torch.float32)), dim=1
            ).numpy()[0]
        p_branches.append(p_tab)
        return float(np.mean(p_branches, axis=0).argmax())

    return _predict_fn, None, None


_PREDICT_FN_BUILDERS = {
    "pytorch_intermediate": _build_predict_fn_intermediate,
    "pytorch_early_fusion": _build_predict_fn_early,
    "sklearn_early_fusion": _build_predict_fn_early,
    "late_fusion_nondp":    _build_predict_fn_late_nondp,
    "late_fusion_deep":     _build_predict_fn_late_deep,
}

# ===========================================================================
# FieldTextNN (copied from run_cf_ablation.py)
# ===========================================================================
class _FieldTextNN(CounterfactualGenerator):

    _accepts_precomputed_batch = True

    def __init__(self, field_name, inner_texnn, X_train_desc):
        self._field_name = field_name
        self._inner       = inner_texnn
        self._train_desc  = np.asarray(X_train_desc, dtype=object).reshape(-1)

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None):
        candidates = self._inner.generate(dataset, sample_idx, target_value=target_value, k=k)
        for cand in candidates:
            src = cand.get("source_train_idx")
            if src is not None:
                cand["text"]       = str(self._train_desc[src])
                cand["text_input"] = self._train_desc[src]
                cand.setdefault("texts", {})["description"] = str(self._train_desc[src])
                cand.setdefault("text_inputs", {})["description"] = self._train_desc[src]
        return candidates

    def generate_batch(self, dataset, sample_indices, model=None, target_value=0, k=None,
                       precomputed=None):
        text_key = ("text", self._field_name)
        if precomputed and text_key in precomputed:
            indices_dict, _ = precomputed[text_key]
            train_text = dataset.get_text_branch(self._field_name, split="train")
            batch = {
                int(idx): TextNN._materialize(
                    indices_dict, int(idx), dataset, train_text, self._field_name,
                    distance_metric_label=self._inner._resolve_distance_metric_label(),
                    text_encoder_label=self._inner._resolve_text_encoder_label(),
                )
                for idx in sample_indices
            }
        else:
            batch = self._inner.generate_batch(dataset, sample_indices, model=model,
                                               target_value=target_value, k=k)
        for candidates in batch.values():
            for cand in candidates:
                src = cand.get("source_train_idx")
                if src is not None:
                    cand["text"]       = str(self._train_desc[src])
                    cand["text_input"] = self._train_desc[src]
                    cand.setdefault("texts", {})["description"] = str(self._train_desc[src])
                    cand.setdefault("text_inputs", {})["description"] = self._train_desc[src]
        return batch


# ===========================================================================
# Factories shared across combos
# ===========================================================================
def _get_embed_fn_for_encoder(encoder: str):
    if encoder == "bert" and _bert_embed_fn is not None:
        return _bert_embed_fn
    if encoder == "word2vec" and _w2v_embed_fn is not None:
        return _w2v_embed_fn
    return _tfidf_embed_fn


def _make_field_lookup_embed_fn(field_name: str, encoder: str):
    lookup  = _field_emb_lookup.get(field_name, {}).get(encoder, {})
    live_fn = _get_embed_fn_for_encoder(encoder)
    def _fn(texts):
        vecs = [None] * len(texts)
        miss_pos, miss_txt = [], []
        for i, t in enumerate(texts):
            v = lookup.get(str(t) if t is not None else "")
            if v is not None:
                vecs[i] = v
            else:
                miss_pos.append(i)
                miss_txt.append(str(t) if t is not None else "")
        if miss_txt:
            live_embs = np.asarray(live_fn(miss_txt), dtype=np.float32)
            for j, pos in enumerate(miss_pos):
                vecs[pos] = live_embs[j]
        return np.stack(vecs)
    return _fn


def _metric_to_static_dist_fn(metric: str):
    if metric == "manhattan":
        return manhattan_distances
    return euclidean_distances


def _make_generators_factory(k, k_search, include_intermediate_fusion,
                              train_latents, test_latents):
    def _factory(tab_cfg, ts_cfg, text_cfg, text_backend_kwargs,
                 image_cfg, image_backend_kwargs):
        encoder    = (text_cfg or {}).get("encoder", "raw")
        metric     = (text_cfg or {}).get("metric", "cosine")
        primary_tab = dataset.primary_tabular_name
        tab_metric = (tab_cfg or {}).get(primary_tab, "euclidean")
        static_dist = _metric_to_static_dist_fn(tab_metric)
        vec_metric  = metric if encoder != "raw" and metric in (
            "cosine", "euclidean", "manhattan"
        ) else "cosine"

        pre_desc    = (_precomputed_by_field["description"].get(encoder)
                       or _precomputed_by_field["description"].get("tfidf"))
        pre_profile = (_precomputed_by_field["company_profile"].get(encoder)
                       or _precomputed_by_field["company_profile"].get("tfidf"))
        pre_reqs    = (_precomputed_by_field["requirements"].get(encoder)
                       or _precomputed_by_field["requirements"].get("tfidf"))
        pre_concat  = (_precomputed_concat.get(encoder)
                       or _precomputed_concat.get("tfidf"))

        extras: dict = {}
        for fname, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS):
            pre = (_precomputed_by_field[fname].get(encoder)
                   or _precomputed_by_field[fname].get("tfidf"))
            inner = TextNN(
                text_name                         = fname,
                k                                 = k,
                text_encoder                      = encoder,
                text_distance_metric              = metric,
                bert_tokenizer                    = text_backend_kwargs.get("bert_tokenizer"),
                bert_model                        = text_backend_kwargs.get("bert_model"),
                bert_device                       = text_backend_kwargs.get("bert_device"),
                word2vec_model                    = text_backend_kwargs.get("word2vec_model"),
                tfidf_vectorizer                  = _tfidf_vec,
                precomputed_train_text_embeddings = pre["train"] if pre else None,
                precomputed_test_text_embeddings  = pre["test"]  if pre else None,
            )
            extras[f"Text_{fname}"] = _FieldTextNN(fname, inner, X_train_description)

        if pre_desc and pre_profile and pre_reqs:
            from run_cf_ablation import MultiFieldFrankensteinNN, MultiFieldCombinedNN
            extras["Frankenstein"] = MultiFieldFrankensteinNN(
                pre_desc=pre_desc, pre_profile=pre_profile, pre_reqs=pre_reqs,
                y_train=dataset.y_train, vec_metric=vec_metric,
                k=k, k_search=k_search, static_dist_fn=static_dist,
            )
            extras["Combined"] = MultiFieldCombinedNN(
                pre_desc=pre_desc, pre_profile=pre_profile, pre_reqs=pre_reqs,
                y_train=dataset.y_train, vec_metric=vec_metric,
                k=k, k_search=k_search, static_dist_fn=static_dist,
            )

        if pre_concat:
            extras["EarlyFusion"] = EarlyFusionNN(
                k=k, distance_metric=vec_metric,
                precomputed_train_text_embeddings=np.concatenate(
                    [pre_concat["train"], dataset.X_train_static], axis=1
                ),
                precomputed_test_text_embeddings=np.concatenate(
                    [pre_concat["test"], dataset.X_test_static], axis=1
                ),
            )

        if include_intermediate_fusion and train_latents is not None:
            from cf_lib.multimodal import IntermediateFusionNN
            extras["IntermediateFusion"] = IntermediateFusionNN(
                k=k, distance_metric=tab_metric,
                precomputed_train_latent=train_latents,
                precomputed_test_latent=test_latents,
            )
        return extras
    return _factory


def _make_objectives_factory(predict_fn_ref):
    def _factory(text_cfg, image_cfg):
        encoder     = (text_cfg or {}).get("encoder", "raw")
        obj_encoder = "tfidf" if encoder == "raw" else encoder

        _Xtest_desc    = np.asarray(X_test_description,     dtype=object).reshape(-1)
        _Xtest_profile = np.asarray(X_test_company_profile, dtype=object).reshape(-1)
        _Xtest_reqs    = np.asarray(X_test_requirements,    dtype=object).reshape(-1)
        _Xtrain_prof   = np.asarray(X_train_company_profile, dtype=object).reshape(-1)
        _Xtrain_reqs   = np.asarray(X_train_requirements,   dtype=object).reshape(-1)

        desc_ctx    = {"embed_fn": _make_field_lookup_embed_fn("description",    obj_encoder)}
        profile_ctx = {"embed_fn": _make_field_lookup_embed_fn("company_profile", obj_encoder)}
        reqs_ctx    = {"embed_fn": _make_field_lookup_embed_fn("requirements",   obj_encoder)}

        def _text_modalities_fn(sample_idx, cand, text_factual):
            cand_desc = str(cand.get("text") or "")
            if "company_profile" in cand:
                cand_profile = str(cand["company_profile"] or "")
                cand_reqs    = str(cand.get("requirements") or "")
            else:
                src = cand.get("source_train_idx")
                if src is not None and src < len(_Xtrain_prof):
                    cand_profile = str(_Xtrain_prof[src])
                    cand_reqs    = str(_Xtrain_reqs[src])
                else:
                    _, cand_profile, cand_reqs = _desc_to_fields.get(
                        cand_desc, (cand_desc, "", "")
                    )
            fact_desc    = str(_Xtest_desc[sample_idx])    if sample_idx < len(_Xtest_desc)    else ""
            fact_profile = str(_Xtest_profile[sample_idx]) if sample_idx < len(_Xtest_profile) else ""
            fact_reqs    = str(_Xtest_reqs[sample_idx])    if sample_idx < len(_Xtest_reqs)    else ""
            return {"text_modalities": {
                "description":     {"candidate": cand_desc,    "factual": fact_desc,    "context": desc_ctx},
                "company_profile": {"candidate": cand_profile, "factual": fact_profile, "context": profile_ctx},
                "requirements":    {"candidate": cand_reqs,    "factual": fact_reqs,    "context": reqs_ctx},
            }}

        return {
            "y_target": target_value,
            "_text_modalities_fn": _text_modalities_fn,
            "plausibility_normalizer": {
                "tab_lof":  _tab_lof,
                "tab_low":  _tab_lof_low,
                "tab_high": _tab_lof_high,
            },
            "tabular_objective_context": {
                "plausibility_normalizer": {
                    "lof":  _tab_lof,
                    "low":  _tab_lof_low,
                    "high": _tab_lof_high,
                }
            },
            "predict_fn": predict_fn_ref[0],
        }
    return _factory


# ===========================================================================
# Main loop — run missing ablations
# ===========================================================================
for strategy, best_key, entry, run_name in todo:
    model_type = entry["model_type"]
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}  |  model: {best_key}  |  run: {run_name}")
    print(f"{'='*60}")

    builder = _PREDICT_FN_BUILDERS.get(model_type)
    if builder is None:
        print(f"  [warn] No predict_fn builder for model_type={model_type} — skipping.")
        continue

    predict_fn, train_latents, test_latents = builder(entry)
    include_if = model_type == "pytorch_intermediate"
    k_search   = min(50, args.k * 5)

    predict_fn_ref = [predict_fn]  # mutable container for closure
    gens_factory   = _make_generators_factory(
        args.k, k_search, include_if, train_latents, test_latents
    )
    obj_factory    = _make_objectives_factory(predict_fn_ref)

    run_distance_ablation(
        dataset=dataset,
        model=None,
        sample_indices=source_indices,
        target_value=target_value,
        k=args.k,
        tab_metrics=tab_metrics,
        ts_metrics=[],
        text_encoders=text_encoders,
        text_vector_metrics=["cosine", "euclidean", "manhattan"],
        text_direct_metrics=["rouge_l", "lcs"],
        text_backend_kwargs=text_bk,
        image_encoders=[],
        image_backend_kwargs={},
        output_dir=args.output_dir,
        run_name=run_name,
        save_full=args.save_full,
        max_combinations=None,
        n_jobs=args.n_jobs,
        objectives_kwargs_factory=obj_factory,
        extra_generators_factory=gens_factory,
    )
    print(f"  Done — results in {Path(args.output_dir) / run_name}/summary.json")
