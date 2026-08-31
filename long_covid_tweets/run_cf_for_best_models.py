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
    cd long_covid_tweets
    python run_cf_for_best_models.py --gpu 7
    python run_cf_for_best_models.py --gpu 7 --no-bert --no-italian-ft
    python run_cf_for_best_models.py --gpu 7 --dry-run   # check only, no ablation
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
# Make cf_lib and examples importable from the long_covid_tweets subdirectory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LocalOutlierFactor

from tweet_cf_factory import build_tweet_dataset
from run_distance_ablation import run_distance_ablation
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import ModalityWisePrototypeSynthesis, MultimodalConsensusRetrieval, EarlyFusionNN
from cf_lib.unimodal import TabularNN
from cf_lib.counterfactual_helpers import find_k_closest_latent

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run missing CF ablations for the best model in each fusion strategy."
)
parser.add_argument("--gpu",              type=int,   default=3)
parser.add_argument("--k",               type=int,   default=20)
parser.add_argument("--max-samples",     type=int,   default=100)
parser.add_argument("--n-jobs",          type=int,   default=1)
parser.add_argument("--output-dir",      type=str,   default="data/ablation_runs")
parser.add_argument("--no-bert",         action="store_true")
parser.add_argument("--italian-ft-path", type=str,   default="data/cc.it.300.bin")
parser.add_argument("--no-italian-ft",   action="store_true")
parser.add_argument("--word2vec-path",   type=str,   default=None)
parser.add_argument("--source-emotion",  type=str,   default="sadness")
parser.add_argument("--target-emotion",  type=str,   default="joy")
parser.add_argument("--save-full",       action="store_true")
parser.add_argument("--dry-run",         action="store_true",
                    help="Print which ablations would be launched without running them.")
parser.add_argument("--strategies",      type=str,   default=None,
                    help="Comma-separated subset of strategies to process "
                         "(e.g. 'early,late'). Default: all three.")
args = parser.parse_args()

DATA_DIR   = Path("data")
OUTPUT_DIR = Path(args.output_dir)
DEVICE     = (
    f"cuda:{args.gpu}"
    if args.gpu is not None and torch.cuda.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------------------------------------------------------------------------
# Load registry
# ---------------------------------------------------------------------------
REGISTRY_PATH = DATA_DIR / "fusion_model_registry.json"
if not REGISTRY_PATH.exists():
    sys.exit(
        f"[error] {REGISTRY_PATH} not found — run evaluate_fusion_models.py first."
    )

with open(REGISTRY_PATH) as fh:
    registry = json.load(fh)

label_classes = registry["label_classes"]
lc_lower      = [c.lower() for c in label_classes]

source_emotion = args.source_emotion.lower()
target_emotion = args.target_emotion.lower()

if source_emotion not in lc_lower:
    sys.exit(f"[error] '{source_emotion}' not in label classes: {label_classes}")
if target_emotion not in lc_lower:
    sys.exit(f"[error] '{target_emotion}' not in label classes: {label_classes}")

source_value = lc_lower.index(source_emotion)
joy_value    = lc_lower.index(target_emotion)

# Strategies to process
all_strategies = ["intermediate", "early", "late"]
if args.strategies:
    all_strategies = [s.strip() for s in args.strategies.split(",")]

# ---------------------------------------------------------------------------
# Determine which ablations need to run
# ---------------------------------------------------------------------------
todo: list[dict] = []   # list of registry model entries that need an ablation

for strategy in all_strategies:
    best_key = registry["best_per_strategy"].get(strategy)
    if best_key is None:
        print(f"[skip] {strategy}: no available model in registry")
        continue
    entry = registry["models"][best_key]
    if not entry.get("available", False):
        print(f"[skip] {strategy}: best model ({best_key}) marked unavailable")
        continue

    run_name = entry["ablation_run_name"] + "_k50"
    summary_path = OUTPUT_DIR / run_name / "summary.json"

    if summary_path.exists():
        print(f"[ok]   {strategy}: ablation already exists → {summary_path}")
    else:
        print(f"[missing] {strategy}: need ablation for {best_key} → {run_name}/")
        todo.append(entry)

if not todo:
    print("\nAll ablations are present. Nothing to do.")
    sys.exit(0)

if args.dry_run:
    print(f"\n[dry-run] Would launch {len(todo)} ablation(s):")
    for e in todo:
        print(f"  {e['ablation_run_name']}  ({e['strategy']}/{e['family']})")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Shared ablation setup (same for all model variants)
# ---------------------------------------------------------------------------
print("\nLoading shared ablation resources …")

produced = build_tweet_dataset(gpu=args.gpu, load_bert=not args.no_bert)
dataset       = produced["dataset"]
text_bk       = produced["text_backend_kwargs"]
y_pred_global = produced["y_pred"]

if y_pred_global is None:
    sys.exit("[error] data/y_pred.npy not found — run evaluate.py first.")

sadness_indices = [int(i) for i, p in enumerate(y_pred_global)
                   if p == source_value]
if args.max_samples is not None:
    sadness_indices = sadness_indices[:args.max_samples]

print(f"Samples predicted '{label_classes[source_value]}': {len(sadness_indices)}")

train_texts = ["" if t is None else str(t) for t in dataset.X_train_text]
test_texts  = ["" if t is None else str(t) for t in dataset.X_test_text]

# TF-IDF (always used, both as text encoder and tfidf fallback)
print("Fitting TF-IDF on training texts …")
_tfidf_vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
_tfidf_vec.fit(train_texts)

def _tfidf_embed_fn(texts):
    return _tfidf_vec.transform(texts).toarray().astype(np.float32)

# Tabular LOF (pre-fitted once; passed via plausibility_normalizer)
print("Pre-fitting tabular LOF …")
_n_lof = max(2, min(20, len(dataset.X_train_static) - 1))
_tab_lof = LocalOutlierFactor(n_neighbors=_n_lof, novelty=True)
_tab_lof.fit(dataset.X_train_static)
_tab_lof_scores = -_tab_lof.score_samples(dataset.X_train_static)
_tab_lof_low    = float(np.percentile(_tab_lof_scores, 5))
_tab_lof_high   = float(np.percentile(_tab_lof_scores, 95))
print(f"  LOF fitted on {len(dataset.X_train_static):,} samples")

# Optional text encoders
text_encoders = ["tfidf", "raw"] if args.no_bert else ["bert", "tfidf", "raw"]

_bert_embed_fn      = None
_italian_ft_embed_fn = None
_w2v_embed_fn        = None

if not args.no_bert:
    from cf_lib.counterfactual_evaluation_helpers import _make_embed_fn_from_e5_kwargs
    _bert_embed_fn = _make_embed_fn_from_e5_kwargs(
        tokenizer=text_bk["bert_tokenizer"],
        model=text_bk["bert_model"],
        device=text_bk["bert_device"],
    )

if args.italian_ft_path and not args.no_italian_ft:
    ft_path = Path(args.italian_ft_path)
    if ft_path.exists():
        print(f"Loading Italian fastText from {ft_path} …")
        try:
            import fasttext as _ft_mod
            _ft_model = _ft_mod.load_model(str(ft_path))
            def _italian_ft_embed_fn(texts, _ft=_ft_model):
                return np.stack([_ft.get_sentence_vector(str(t).replace("\n", " ")).astype(np.float32)
                                 for t in texts])
        except ImportError:
            from gensim.models.fasttext import load_facebook_vectors
            _ft_kv = load_facebook_vectors(str(ft_path))
            def _italian_ft_embed_fn(texts, _kv=_ft_kv):
                out = []
                for t in texts:
                    toks = str(t).lower().split()
                    vecs = [_kv.get_vector(w) for w in toks if w]
                    out.append(np.mean(vecs, 0).astype(np.float32) if vecs
                                else np.zeros(_kv.vector_size, dtype=np.float32))
                return np.stack(out)
        text_bk["text_embed_fn"] = _italian_ft_embed_fn
        text_encoders.insert(-1, "custom")
        print("  Italian fastText loaded.")

if args.word2vec_path:
    wv_path = Path(args.word2vec_path)
    if wv_path.exists():
        from gensim.models import KeyedVectors
        _w2v_kv = (KeyedVectors.load(str(wv_path))
                   if str(wv_path).endswith(".kv")
                   else KeyedVectors.load_word2vec_format(str(wv_path),
                        binary=str(wv_path).endswith(".bin")))
        def _w2v_embed_fn(texts, _kv=_w2v_kv):
            out = []
            for t in texts:
                toks = str(t).lower().split()
                vecs = [_kv[w] for w in toks if w in _kv]
                out.append(np.mean(vecs, 0).astype(np.float32) if vecs
                            else np.zeros(_kv.vector_size, dtype=np.float32))
            return np.stack(out)
        text_bk["word2vec_model"] = _w2v_kv
        text_encoders.insert(-1, "word2vec")

# Precompute full-corpus embeddings once per encoder (reused across combos)
def _precompute_once(name, embed_fn):
    all_texts = train_texts + test_texts
    print(f"  Precomputing {name} embeddings for {len(all_texts)} texts …")
    embs = np.asarray(embed_fn(all_texts), dtype=np.float32)
    return {"train": embs[:len(train_texts)].copy(),
            "test":  embs[len(train_texts):].copy()}

precomputed: Dict[str, dict] = {"tfidf": _precompute_once("tfidf", _tfidf_embed_fn)}
if _bert_embed_fn is not None:
    precomputed["bert"]   = _precompute_once("bert",   _bert_embed_fn)
if _italian_ft_embed_fn is not None:
    precomputed["custom"] = _precompute_once("custom", _italian_ft_embed_fn)
if _w2v_embed_fn is not None:
    precomputed["word2vec"] = _precompute_once("word2vec", _w2v_embed_fn)

text_bk["precomputed_text_embeddings_by_encoder"] = precomputed

# Build text-string → embedding lookup (used by non-transformer predict_fns
# to avoid live inference for Frankenstein/Combined candidates)
# All candidate texts are real training samples, so this lookup always hits.
print("Building text→CLS lookup …")
_text_to_cls: Dict[str, np.ndarray] = {}
if EMB_CACHE_PATH := DATA_DIR / "early_fusion_text_embeddings.pt":
    if EMB_CACHE_PATH.exists():
        _ec = torch.load(EMB_CACHE_PATH, map_location="cpu")
        _all_cls = torch.cat([_ec["train"], _ec["test"]], dim=0).numpy().astype("float32")
        for _txt, _emb in zip(train_texts + test_texts, _all_cls):
            _text_to_cls[_txt] = _emb
        print(f"  CLS lookup built: {len(_text_to_cls):,} entries")


# ---------------------------------------------------------------------------
# Shared helper: per-combo generator and objectives factories
# (identical across all model variants; predict_fn injected per-run)
# ---------------------------------------------------------------------------
tab_metrics         = ["euclidean", "manhattan"]
text_vector_metrics = ["cosine", "euclidean", "manhattan"]
text_direct_metrics = ["rouge_l", "lcs"]
_k_search           = min(50, args.k * 5)


def _get_embed_fn(encoder: str):
    if encoder == "bert"     and _bert_embed_fn      is not None: return _bert_embed_fn
    if encoder == "custom"   and _italian_ft_embed_fn is not None: return _italian_ft_embed_fn
    if encoder == "word2vec" and _w2v_embed_fn        is not None: return _w2v_embed_fn
    return _tfidf_embed_fn


def _metric_to_dist_fn(metric: str):
    if metric == "manhattan": return manhattan_distances
    return euclidean_distances


def _make_generators_factory(include_intermediate_fusion: bool,
                              train_latents=None, test_latents=None):
    """Return an extra_generators_factory closure for the ablation runner."""

    def _factory(tab_cfg, ts_cfg, text_cfg, text_backend_kwargs,
                 image_cfg, image_backend_kwargs):
        encoder  = (text_cfg or {}).get("encoder", "raw")
        tab_m    = (tab_cfg  or {}).get(dataset.primary_tabular_name, "euclidean")
        embed_fn = _get_embed_fn(encoder)
        pc       = precomputed.get(encoder if encoder != "raw" else "tfidf")
        dist_fn  = _metric_to_dist_fn(tab_m)

        extras = {
            "MPS": ModalityWisePrototypeSynthesis(
                k=args.k, k_search=_k_search,
                static_dist_fn=dist_fn, e5_embed_fn=embed_fn,
            ),
            "MC-R": MultimodalConsensusRetrieval(
                k=args.k, k_search=_k_search,
                static_dist_fn=dist_fn, e5_embed_fn=embed_fn,
            ),
            "EarlyFusion": EarlyFusionNN(
                k=args.k, distance_metric=tab_m, e5_embed_fn=embed_fn,
                precomputed_train_text_embeddings=None if pc is None else pc["train"],
                precomputed_test_text_embeddings=None  if pc is None else pc["test"],
            ),
        }

        if include_intermediate_fusion and train_latents is not None:
            class _LatentNN(CounterfactualGenerator):
                def __init__(self):
                    self.k              = args.k
                    self.distance_metric = tab_m
                def generate(self, ds, sample_idx, model=None, target_value=0, k=None):
                    k = k or self.k
                    indices, _ = find_k_closest_latent(
                        X_train_latent=train_latents,
                        y_train=ds.y_train,
                        X_test_latent=test_latents,
                        selected_test_indices=[sample_idx],
                        target_value=target_value,
                        k=k,
                        distance_metric=self.distance_metric,
                    )
                    return TabularNN._materialize(
                        indices, sample_idx, ds,
                        distance_metric_label=self.distance_metric or "euclidean",
                    )
            extras["IntermediateFusion"] = _LatentNN()

        return extras

    return _factory


def _make_objectives_factory(predict_fn_ref: list):
    """Return an objectives_kwargs_factory closure.

    predict_fn_ref is a 1-element list so the closure captures the reference
    and we can update it after building.
    """
    def _factory(text_cfg, image_cfg):
        encoder     = (text_cfg or {}).get("encoder", "raw")
        obj_encoder = "tfidf" if encoder == "raw" else encoder
        embed_fn    = _get_embed_fn(obj_encoder)
        kwargs      = {
            "text_objective_context": {"embed_fn": embed_fn},
            "y_target":               joy_value,
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
        }
        if predict_fn_ref[0] is not None:
            kwargs["predict_fn"] = predict_fn_ref[0]
        return kwargs
    return _factory


# ---------------------------------------------------------------------------
# Inline model class definitions (same as evaluate_fusion_models.py)
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
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim
    def forward(self, x): return self.net(x)


class _MultimodalClassifier(nn.Module):
    _TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"
    def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
        super().__init__()
        from transformers import AutoModel
        self.text_encoder = AutoModel.from_pretrained(self._TEXT_MODEL)
        enc_dim           = self.text_encoder.config.hidden_size
        self.text_proj    = nn.Sequential(
            nn.Linear(enc_dim, text_hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.tab_head     = _TabHead(d_tab, tab_hidden_dims, dropout)
        self.classifier   = nn.Linear(text_hidden_dim + self.tab_head.out_dim, n_classes)
    def forward(self, ids, mask, X_static):
        cls = self.text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.classifier(torch.cat([self.text_proj(cls), self.tab_head(X_static)], 1))


# ---------------------------------------------------------------------------
# Per-model-type predict_fn builders
# ---------------------------------------------------------------------------

def _build_predict_fn_intermediate(entry: dict):
    """PyTorch MultimodalClassifier + precomputed training-sample cache."""
    from transformers import AutoTokenizer
    ckpt  = torch.load(entry["model_files"]["main"], map_location=DEVICE)
    tok   = text_bk.get("bert_tokenizer") or AutoTokenizer.from_pretrained(
        _MultimodalClassifier._TEXT_MODEL)
    model = _MultimodalClassifier(
        d_tab           = len(ckpt["tab_cols"]),
        n_classes       = len(label_classes),
        tab_hidden_dims = ckpt["config"]["tab_hidden_dims"],
        text_hidden_dim = ckpt["config"]["text_hidden_dim"],
        dropout         = ckpt["config"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Latent extraction for IntermediateFusion generator
    _latent_buf = {}
    def _hook(module, inp, out): _latent_buf["z"] = inp[0].detach().cpu().numpy()
    _handle = model.classifier.register_forward_hook(_hook)

    n_tr     = len(dataset.X_train_static)
    tr_parts, te_parts = [], []
    print("  Extracting intermediate-fusion latents …")
    with torch.no_grad():
        for split_texts, split_static, bucket in [
            (train_texts, dataset.X_train_static, tr_parts),
            (test_texts,  dataset.X_test_static,  te_parts),
        ]:
            for i in range(0, len(split_texts), 32):
                enc = tok(split_texts[i:i+32], max_length=128,
                          padding="max_length", truncation=True, return_tensors="pt")
                tab = torch.tensor(split_static[i:i+32].astype(np.float32)).to(DEVICE)
                model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), tab)
                bucket.append(_latent_buf["z"].copy())
    _handle.remove()
    train_latents = np.vstack(tr_parts).astype(np.float32)
    test_latents  = np.vstack(te_parts).astype(np.float32)
    print(f"  Latent shape: {train_latents.shape[1]}-dim")

    # Training-sample prediction cache
    print("  Building training-sample prediction cache …")
    _cache: dict = {}
    _lock = threading.Lock()
    with torch.no_grad():
        for i in range(0, n_tr, 64):
            enc = tok([str(t) for t in dataset.X_train_text[i:i+64]],
                      max_length=128, padding="max_length",
                      truncation=True, return_tensors="pt")
            tab = torch.tensor(dataset.X_train_static[i:i+64].astype(np.float32)).to(DEVICE)
            logits = model(enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), tab)
            for j, pred in enumerate(logits.argmax(1).cpu().numpy()):
                _cache[(
                    dataset.X_train_static[i+j].astype(np.float32).tobytes(),
                    str(dataset.X_train_text[i+j]),
                )] = float(pred)
    print(f"  Cached {len(_cache):,} training predictions")

    def _predict_fn(x_tab, x_ts, text_candidate):
        key = (np.asarray(x_tab, np.float32).tobytes(),
               str(text_candidate) if text_candidate is not None else "")
        hit = _cache.get(key)
        if hit is not None: return hit
        text = str(text_candidate) if text_candidate is not None else ""
        with _lock:
            enc = tok([text], max_length=128, padding="max_length",
                      truncation=True, return_tensors="pt")
            tab = torch.tensor(np.asarray(x_tab, np.float32)[None], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = model(enc["input_ids"].to(DEVICE),
                               enc["attention_mask"].to(DEVICE), tab)
        return float(logits.argmax(1).cpu().item())

    return _predict_fn, train_latents, test_latents


def _build_predict_fn_early(entry: dict):
    """Early-fusion predict_fn: concat CLS(text) + x_tab → model."""
    model_type = entry["model_type"]   # pytorch_early_fusion or sklearn_early_fusion
    mf         = entry["model_files"]["main"]

    if model_type == "pytorch_early_fusion":
        ckpt = torch.load(mf, map_location="cpu")
        model = _EarlyFusionMLP(ckpt["d_in"], ckpt["n_classes"],
                                 ckpt["hidden"], ckpt["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        def _forward(x_ef):
            with torch.no_grad():
                return int(model(torch.tensor(x_ef[None].astype("float32"))).argmax(1).item())
    else:  # sklearn
        with open(mf, "rb") as fh:
            bundle = pickle.load(fh)
        sk_model = bundle if not isinstance(bundle, dict) else bundle["model"]
        def _forward(x_ef):
            return int(sk_model.predict(x_ef[None])[0])

    def _predict_fn(x_tab, x_ts, text_candidate):
        text_str = str(text_candidate) if text_candidate is not None else ""
        cls_emb  = _text_to_cls.get(text_str)
        if cls_emb is None:
            # Fallback: return 0 (unknown text — should not occur in practice since
            # all candidates are retrieved from the training set)
            return 0.0
        x_ef = np.concatenate([cls_emb, np.asarray(x_tab, np.float32)])
        return float(_forward(x_ef))

    return _predict_fn, None, None   # no latents for early fusion


def _build_predict_fn_late_nondp(entry: dict):
    """Late-fusion non-deep: avg(tfidf_logreg proba, tabular sklearn proba)."""
    with open(entry["model_files"]["text"], "rb") as fh:
        bundle_t = pickle.load(fh)
    tfidf_vec = bundle_t["tfidf"]
    logreg    = bundle_t["model"]

    with open(entry["model_files"]["tabular"], "rb") as fh:
        bundle_s = pickle.load(fh)
    tab_model = bundle_s if not isinstance(bundle_s, dict) else bundle_s["model"]

    def _predict_fn(x_tab, x_ts, text_candidate):
        text = str(text_candidate) if text_candidate is not None else ""
        p_txt = logreg.predict_proba(tfidf_vec.transform([text])).astype("float32")[0]
        p_tab = tab_model.predict_proba(
            np.asarray(x_tab, np.float32)[None]
        ).astype("float32")[0]
        return float((0.5 * p_txt + 0.5 * p_tab).argmax())

    return _predict_fn, None, None


def _build_predict_fn_late_deep(entry: dict):
    """Late-fusion deep: avg(TextMLP(CLS) proba, TabMLP(x_tab) proba)."""
    ckpt_t  = torch.load(entry["model_files"]["text"],    map_location="cpu")
    ckpt_tb = torch.load(entry["model_files"]["tabular"], map_location="cpu")

    d_t  = ckpt_t["state_dict"]["net.0.weight"].shape[1]
    h_t  = ckpt_t["state_dict"]["net.0.weight"].shape[0]
    nc   = ckpt_t["state_dict"]["net.2.weight"].shape[0]
    txt_mlp = _TextMLP(d_t, nc, h_t)
    txt_mlp.load_state_dict(ckpt_t["state_dict"])
    txt_mlp.eval()

    d_tb = ckpt_tb["state_dict"]["net.0.weight"].shape[1]
    h_tb = ckpt_tb["state_dict"]["net.0.weight"].shape[0]
    nc2  = ckpt_tb["state_dict"]["net.4.weight"].shape[0]
    tab_mlp = _TabMLP(d_tb, nc2, h_tb)
    tab_mlp.load_state_dict(ckpt_tb["state_dict"])
    tab_mlp.eval()

    def _predict_fn(x_tab, x_ts, text_candidate):
        text_str = str(text_candidate) if text_candidate is not None else ""
        cls_emb  = _text_to_cls.get(text_str)
        if cls_emb is None:
            return 0.0
        with torch.no_grad():
            p_txt = torch.softmax(
                txt_mlp(torch.tensor(cls_emb[None])), dim=1
            ).numpy().astype("float32")[0]
            p_tab = torch.softmax(
                tab_mlp(torch.tensor(np.asarray(x_tab, np.float32)[None])), dim=1
            ).numpy().astype("float32")[0]
        return float((0.5 * p_txt + 0.5 * p_tab).argmax())

    return _predict_fn, None, None


_PREDICT_FN_BUILDERS = {
    "pytorch_intermediate": _build_predict_fn_intermediate,
    "pytorch_early_fusion": _build_predict_fn_early,
    "sklearn_early_fusion": _build_predict_fn_early,
    "late_fusion_nondp":    _build_predict_fn_late_nondp,
    "late_fusion_deep":     _build_predict_fn_late_deep,
}


# ---------------------------------------------------------------------------
# Main loop: run ablation for each pending entry
# ---------------------------------------------------------------------------
for entry in todo:
    strategy      = entry["strategy"]
    family        = entry["family"]
    run_name      = entry["ablation_run_name"]
    model_type    = entry["model_type"]

    print(f"\n{'='*65}")
    print(f"Running ablation: {run_name}  ({strategy}/{family})")
    print(f"{'='*65}")

    # Build predict_fn (and optional latents)
    builder = _PREDICT_FN_BUILDERS.get(model_type)
    if builder is None:
        print(f"[warn] No predict_fn builder for model_type={model_type!r} — skipping")
        continue

    predict_fn, train_latents, test_latents = builder(entry)

    # IntermediateFusion generator only valid for the intermediate-fusion deep model
    has_latents = train_latents is not None

    # Wrap in a mutable reference so the factory closure can access it
    predict_fn_ref = [predict_fn]

    run_distance_ablation(
        dataset                   = dataset,
        model                     = None,
        sample_indices            = sadness_indices,
        target_value              = joy_value,
        k                         = args.k,
        tab_metrics               = tab_metrics,
        ts_metrics                = [],
        text_encoders             = text_encoders,
        text_vector_metrics       = text_vector_metrics,
        text_direct_metrics       = text_direct_metrics,
        text_backend_kwargs       = text_bk,
        output_dir                = str(OUTPUT_DIR),
        run_name                  = run_name,
        save_full                 = args.save_full,
        n_jobs                    = args.n_jobs,
        objectives_kwargs_factory = _make_objectives_factory(predict_fn_ref),
        extra_generators_factory  = _make_generators_factory(
            include_intermediate_fusion=has_latents,
            train_latents=train_latents,
            test_latents=test_latents,
        ),
    )

    print(f"\n[done] {run_name}")

print("\nAll pending ablations complete.")
