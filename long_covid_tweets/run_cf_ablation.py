"""Counterfactual ablation: sadness → joy on Long COVID tweets.

Finds all test samples predicted as "sadness", then sweeps distance metric
combinations via cf_lib's run_distance_ablation to generate counterfactuals
that flip each sample to "joy".

For each combo the same text encoder used to *search* for counterfactuals is
also used to *evaluate* proximity/plausibility objectives.  When the search
encoder is "raw" (direct string metrics), tfidf is used as the evaluation
embedding instead.

Run
---
    cd long_covid_tweets
    python run_cf_ablation.py --gpu 7

    # Smoke test (3 samples, no bert, Italian fastText + tfidf + raw)
    python run_cf_ablation.py --gpu 7 --max-samples 3 --no-bert \\
        --italian-ft-path data/cc.it.300.bin

    # With word2vec as well
    python run_cf_ablation.py --gpu 7 --max-samples 3 --no-bert \\
        --italian-ft-path data/cc.it.300.bin --word2vec-path data/glove_twitter_25.kv

    # Without word2vec or fastText
    python run_cf_ablation.py --gpu 7 --max-samples 3 --no-bert --no-italian-ft

Outputs
-------
    data/ablation_runs/<timestamp>/
        summary.jsonl   — one line per metric combination (streamed)
        summary.json    — full payload at the end
        combo_NNNNN_results.pkl  — full candidate dicts (when --save-full)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import threading
from typing import Dict, Optional

import numpy as np

# Make cf_lib and examples importable from the long_covid_tweets subdirectory
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from tweet_cf_factory import build_tweet_dataset
from run_distance_ablation import run_distance_ablation  # from examples/
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import ModalityWisePrototypeSynthesis, MultimodalConsensusRetrieval, EarlyFusionNN
from cf_lib.unimodal import TabularNN
from cf_lib.counterfactual_helpers import find_k_closest_latent
from cf_lib.counterfactual_evaluation_helpers import fit_proximity_normalizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Sadness→Joy counterfactual ablation on Long COVID tweets."
)
parser.add_argument("--gpu",              type=int,   default=7)
parser.add_argument("--k",               type=int,   default=50,
                    help="Number of nearest neighbours per generator (default: 20).")
parser.add_argument("--max-samples",     type=int,   default=None,
                    help="Cap on how many sadness samples to process (default: all).")
parser.add_argument("--max-combinations",type=int,   default=None,
                    help="Cap on metric combinations to try (default: all).")
parser.add_argument("--n-jobs",          type=int,   default=1,
                    help="Number of ablation combinations to run in parallel.")
parser.add_argument("--output-dir",      type=str,   default="data/ablation_runs")
parser.add_argument("--run-name",        type=str,   default=None)
parser.add_argument("--k-search-combined", type=int, default=300,
                    help="k_search pool size for ModalityWisePrototypeSynthesis and MultimodalConsensusRetrieval (default: 300).")
parser.add_argument("--fusion-strategy", type=str,   default="intermediate",
                    choices=["intermediate", "early", "late"],
                    help="Which fusion model to use for predict_fn / latents. "
                         "intermediate=best_model.pt (default); "
                         "early=best_model_early_fusion_mlp.pt; "
                         "late=tfidf_logreg_text + gbt_tabular averaged.")
parser.add_argument("--save-full",       action="store_true", default=True,
                    help="Save full candidate dicts as .pkl per combination.")
parser.add_argument("--no-bert",         action="store_true",
                    help="Skip loading the BERT text backend (faster for smoke tests).")
parser.add_argument("--word2vec-path",   type=str,   default=None,
                    help="Path to a gensim word2vec .bin or .kv file. "
                         "Enables word2vec as a search and evaluation encoder.")
parser.add_argument("--italian-ft-path", type=str,   default="data/cc.it.300.bin",
                    help="Path to a fastText .bin model (e.g. cc.it.300.bin). "
                         "Used as 'custom' encoder for both NN search and objectives.")
parser.add_argument("--no-italian-ft",   action="store_true",
                    help="Skip loading the Italian fastText model even if --italian-ft-path exists.")
parser.add_argument("--source-emotion",  type=str,   default="sadness",
                    help="Emotion class to explain (default: sadness).")
parser.add_argument("--target-emotion",  type=str,   default="joy",
                    help="Counterfactual target emotion (default: joy).")
parser.add_argument(
    "--eval-pkls",
    type=str,
    default=None,
    metavar="DIR[,DIR,...]",
    help=(
        "Instead of generating new counterfactuals, re-evaluate saved "
        "combo_*_results.pkl files at multiple k values. "
        "Comma-separated list of run directories (each containing pkls + "
        "summary.jsonl). Results are written to <first-dir>/k_ablation_summary.json."
    ),
)
parser.add_argument(
    "--k-values",
    type=str,
    default="1,5,10,20,50",
    help="Comma-separated k values for --eval-pkls mode (default: 1,5,10,20,50).",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load dataset and predictions
# ---------------------------------------------------------------------------
produced = build_tweet_dataset(gpu=args.gpu, load_bert=not args.no_bert,
                               fusion_strategy=args.fusion_strategy)

dataset       = produced["dataset"]
text_bk       = produced["text_backend_kwargs"]
label_classes = produced["label_classes"]
y_pred        = produced["y_pred"]

if y_pred is None:
    raise FileNotFoundError(
        "data/y_pred.npy not found — run evaluate.py first."
    )

# ---------------------------------------------------------------------------
# Resolve sadness / joy indices
# ---------------------------------------------------------------------------
lc_lower      = [c.lower() for c in label_classes]
source_emotion = args.source_emotion.lower()
target_emotion = args.target_emotion.lower()

if source_emotion not in lc_lower:
    raise ValueError(
        f"'{source_emotion}' not found in label classes: {label_classes}."
    )
if target_emotion not in lc_lower:
    raise ValueError(
        f"'{target_emotion}' not found in label classes: {label_classes}."
    )
if source_emotion == target_emotion:
    raise ValueError("--source-emotion and --target-emotion must be different.")

source_value = lc_lower.index(source_emotion)
joy_value    = lc_lower.index(target_emotion)

sadness_indices = [int(i) for i, p in enumerate(y_pred) if p == source_value]
if args.max_samples is not None:
    sadness_indices = sadness_indices[: args.max_samples]

print(f"\nLabel classes : {label_classes}")
print(f"source index  : {source_value} ({label_classes[source_value]})  |  "
      f"target index: {joy_value} ({label_classes[joy_value]})")
print(f"Test samples predicted as '{label_classes[source_value]}': {len(sadness_indices)}")
if len(sadness_indices) == 0:
    raise RuntimeError(
        f"No test samples predicted as '{label_classes[source_value]}' — nothing to explain."
    )

# ---------------------------------------------------------------------------
# Text encoder / metric options
# ---------------------------------------------------------------------------
text_encoders       = ["bert", "tfidf", "raw"] if not args.no_bert else ["tfidf", "raw"]
text_vector_metrics = ["cosine", "euclidean", "manhattan"]
text_direct_metrics = ["rouge_l", "lcs"]
tab_metrics         = ["euclidean", "manhattan"]

# ---------------------------------------------------------------------------
# Build embed_fn for each text encoder (used for objective evaluation)
# ---------------------------------------------------------------------------
train_texts = ["" if t is None else str(t) for t in dataset.X_train_text]
test_texts = ["" if t is None else str(t) for t in dataset.X_test_text]

# — tfidf (always available) —
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nFitting TF-IDF on training texts …")
_tfidf_vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
_tfidf_vec.fit(train_texts)

def _tfidf_embed_fn(texts):
    # Raw TF-IDF vectors — the search pipeline normalises for cosine only.
    return _tfidf_vec.transform(texts).toarray().astype(np.float32)

# — tabular LOF (pre-fitted once; reused across all combos/candidates) —
# tabular_plausibility() fits a fresh LOF on the full training set per call,
# which takes ~1.5 s × 12 000 candidates/combo → hours of wall time.
# Pre-fitting once and passing via plausibility_normalizer reduces this to a
# cheap score_samples() lookup (read-only, thread-safe).
print("Pre-fitting tabular LOF on training set …")
from sklearn.neighbors import LocalOutlierFactor as _LOF
_n_lof_neighbors = max(2, min(20, len(dataset.X_train_static) - 1))
_tab_lof = _LOF(n_neighbors=_n_lof_neighbors, novelty=True)
_tab_lof.fit(dataset.X_train_static)
_tab_lof_train_scores = -_tab_lof.score_samples(dataset.X_train_static)
_tab_lof_low  = float(np.percentile(_tab_lof_train_scores, 5))
_tab_lof_high = float(np.percentile(_tab_lof_train_scores, 95))
print(f"  LOF fitted on {len(dataset.X_train_static):,} samples "
      f"(score range [{_tab_lof_low:.3f}, {_tab_lof_high:.3f}])")
print("Pre-computing proximity reference distribution …")
_prox_norm = fit_proximity_normalizer(
    X_tab_obs=dataset.X_train_static,
    y_train=dataset.y_train,
)
print(f"  Proximity ref dists: {len(_prox_norm.get('tab_ref_dists', []))} tabular pairs sampled.")

# ---------------------------------------------------------------------------
# PyTorch model latent space (for IntermediateFusion generator)
# ---------------------------------------------------------------------------
class PyTorchLatentNN(CounterfactualGenerator):
    """Nearest-neighbour search in the PyTorch classifier's penultimate-layer space.

    Equivalent to IntermediateFusionNN but uses precomputed latents extracted
    via a forward hook, bypassing the Keras-specific latent model API.
    """

    def __init__(self, k: int, train_latents: np.ndarray, test_latents: np.ndarray,
                 distance_metric: str = "euclidean"):
        self.k = k
        self.train_latents = train_latents
        self.test_latents  = test_latents
        self.distance_metric = distance_metric

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None):
        k = k if k is not None else self.k
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
            distance_metric_label=self.distance_metric or "euclidean",
        )


def _compute_torch_latents(torch_model, dataset, tokenizer, device, batch_size=32):
    """Extract penultimate-layer (fused) representations from the PyTorch classifier.

    Registers a forward hook on ``model.classifier`` to capture the concatenated
    tabular+text embedding (shape: n × (tab_out_dim + text_hidden_dim)).

    Returns (train_latents, test_latents) as float32 numpy arrays.
    """
    import torch

    torch_model.eval()
    _buf: dict = {}

    def _hook(module, inp, out):
        _buf["z"] = inp[0].detach().cpu().numpy()

    handle = torch_model.classifier.register_forward_hook(_hook)

    def _run(X_static, X_text):
        n = len(X_static)
        rows = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            texts = [str(t) for t in X_text[start:end]]
            enc   = tokenizer(texts, max_length=128, padding="max_length",
                              truncation=True, return_tensors="pt")
            tab   = torch.tensor(X_static[start:end], dtype=torch.float32).to(device)
            with torch.no_grad():
                # forward(input_ids, attention_mask, X_static)
                torch_model(enc["input_ids"].to(device),
                            enc["attention_mask"].to(device),
                            tab)
            rows.append(_buf["z"].copy())
        return np.vstack(rows).astype(np.float32)

    print("Extracting model latents for train split …")
    train_latents = _run(dataset.X_train_static, dataset.X_train_text)
    print("Extracting model latents for test split …")
    test_latents  = _run(dataset.X_test_static,  dataset.X_test_text)
    handle.remove()
    print(f"  Latent shape: {train_latents.shape[1]}-dim")
    return train_latents, test_latents


# Load the PyTorch classifier and precompute latents (once, reused across all combos)
_torch_latents: Optional[tuple] = None
_predict_fn: Optional[object] = None
_predict_fn_lock = threading.Lock()  # serializes live GPU inference (cache-miss path)
_DATA_DIR = Path(__file__).parent / "data"
_fusion_strategy = args.fusion_strategy
_pt_path  = _DATA_DIR / "best_model.pt"

if _fusion_strategy == "intermediate" and _pt_path.exists():
    try:
        import torch
        import torch.nn as _nn
        from transformers import AutoModel as _AutoModel

        # Inline definition to avoid importing hparam_search.py (which has
        # module-level argparse that would collide with our own CLI args).
        _TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"

        class _TabHead(_nn.Module):
            def __init__(self, d_in, hidden_dims, dropout):
                super().__init__()
                layers, in_dim = [], d_in
                for h in hidden_dims:
                    layers += [_nn.Linear(in_dim, h), _nn.ReLU(), _nn.Dropout(dropout)]
                    in_dim = h
                self.net = _nn.Sequential(*layers)
                self.out_dim = in_dim
            def forward(self, x):
                return self.net(x)

        class _MultimodalClassifier(_nn.Module):
            def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
                super().__init__()
                self.text_encoder = _AutoModel.from_pretrained(_TEXT_MODEL)
                enc_dim = self.text_encoder.config.hidden_size
                self.text_proj = _nn.Sequential(
                    _nn.Linear(enc_dim, text_hidden_dim), _nn.ReLU(), _nn.Dropout(dropout)
                )
                self.tab_head   = _TabHead(d_tab, tab_hidden_dims, dropout)
                self.classifier = _nn.Linear(text_hidden_dim + self.tab_head.out_dim, n_classes)
            def forward(self, input_ids, attention_mask, X_static):
                cls  = self.text_encoder(input_ids=input_ids,
                                         attention_mask=attention_mask).last_hidden_state[:, 0]
                return self.classifier(
                    torch.cat([self.text_proj(cls), self.tab_head(X_static)], dim=1)
                )

        _device_latent = (
            f"cuda:{args.gpu}"
            if args.gpu is not None and torch.cuda.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _ckpt = torch.load(_pt_path, map_location=_device_latent)
        _torch_model = _MultimodalClassifier(
            d_tab           = len(_ckpt["tab_cols"]),
            n_classes       = len(_ckpt["label_classes"]),
            tab_hidden_dims = _ckpt["config"]["tab_hidden_dims"],
            text_hidden_dim = _ckpt["config"]["text_hidden_dim"],
            dropout         = _ckpt["config"]["dropout"],
        ).to(_device_latent)
        _torch_model.load_state_dict(_ckpt["state_dict"])
        _torch_model.eval()

        # Tokenizer: reuse from bert backend if available, else load fresh
        _latent_tokenizer = text_bk.get("bert_tokenizer") or __import__(
            "transformers", fromlist=["AutoTokenizer"]
        ).AutoTokenizer.from_pretrained("cardiffnlp/twitter-xlm-roberta-base")

        _torch_latents = _compute_torch_latents(
            _torch_model, dataset, _latent_tokenizer, _device_latent
        )

        # Precompute model predictions for all training samples in one batched pass.
        # Unimodal / Combined / EarlyFusion / IntermediateFusion candidates are pure
        # training samples, so their predictions are exact cache hits.  Only
        # ModalityWisePrototypeSynthesis assembles hybrids (tabular from one training sample, text
        # from another), so those candidates will miss and fall through to serialized
        # live inference.
        print("Precomputing model predictions for all training samples …")
        _n_tr = len(dataset.X_train_static)
        _pred_rows = []
        for _bs in range(0, _n_tr, 64):
            _be = min(_bs + 64, _n_tr)
            _ptexts = [str(t) for t in dataset.X_train_text[_bs:_be]]
            _penc = _latent_tokenizer(
                _ptexts, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            _ptab = torch.tensor(
                dataset.X_train_static[_bs:_be].astype(np.float32),
                dtype=torch.float32,
            ).to(_device_latent)
            with torch.no_grad():
                _plogits = _torch_model(
                    _penc["input_ids"].to(_device_latent),
                    _penc["attention_mask"].to(_device_latent),
                    _ptab,
                )
            _pred_rows.append(_plogits.argmax(dim=-1).cpu().numpy().astype(np.float32))
        _train_pred_cache: dict = {
            (dataset.X_train_static[_i].astype(np.float32).tobytes(),
             str(dataset.X_train_text[_i])): float(np.concatenate(_pred_rows)[_i])
            for _i in range(_n_tr)
        }
        print(f"  Cached {len(_train_pred_cache):,} training-sample predictions.")

        # Build predict_fn for outcome evaluation.
        # Returns the predicted class index (float) so outcome_objective = 0.0
        # when the counterfactual is classified as the target class.
        # Fast path: cache lookup (no GPU call).
        # Slow path: serialized live inference for Frankenstein / Combined hybrids.
        def _predict_fn(x_tab, x_ts, text_candidate,
                        _model=_torch_model, _tok=_latent_tokenizer, _dev=_device_latent,
                        _cache=_train_pred_cache, _lock=_predict_fn_lock):
            if x_tab is not None:
                _key = (np.asarray(x_tab, dtype=np.float32).tobytes(),
                        str(text_candidate) if text_candidate is not None else "")
                _hit = _cache.get(_key)
                if _hit is not None:
                    return _hit
            # Cache miss: Frankenstein / Combined hybrid — serialized live inference.
            import torch as _t
            text = str(text_candidate) if text_candidate is not None else ""
            with _lock:
                enc = _tok([text], max_length=128, padding="max_length",
                           truncation=True, return_tensors="pt")
                tab = _t.tensor(np.asarray(x_tab, dtype=np.float32)[None, :],
                                dtype=_t.float32).to(_dev)
                with _t.no_grad():
                    logits = _model(enc["input_ids"].to(_dev),
                                    enc["attention_mask"].to(_dev), tab)
            return float(logits.argmax(dim=-1).cpu().item())

    except Exception as _e:
        print(f"[warn] Could not compute model latents — IntermediateFusion will be skipped: {_e}")
elif _fusion_strategy == "intermediate":
    print(f"[warn] best_model.pt not found at {_pt_path} — IntermediateFusion will be skipped.")

# ---- Early fusion predict_fn (MLP on frozen XLM-R CLS + tabular) -----------
if _fusion_strategy == "early":
    _ef_path = _DATA_DIR / "best_model_early_fusion_mlp.pt"
    _ef_emb_path = _DATA_DIR / "early_fusion_text_embeddings.pt"
    if _ef_path.exists() and _ef_emb_path.exists():
        try:
            import torch as _t
            import torch.nn as _nn

            class _EarlyFusionMLP(_nn.Module):
                def __init__(self, d_in, n_classes, hidden, dropout):
                    super().__init__()
                    self.net = _nn.Sequential(
                        _nn.Linear(d_in, hidden), _nn.ReLU(), _nn.Dropout(dropout),
                        _nn.Linear(hidden, hidden // 2), _nn.ReLU(), _nn.Dropout(dropout),
                        _nn.Linear(hidden // 2, n_classes),
                    )
                def forward(self, x): return self.net(x)

            _ef_ckpt = _t.load(_ef_path, map_location="cpu")
            _ef_dev  = (f"cuda:{args.gpu}" if args.gpu is not None and _t.cuda.is_available()
                        else ("cuda" if _t.cuda.is_available() else "cpu"))
            _ef_model = _EarlyFusionMLP(
                _ef_ckpt["d_in"], _ef_ckpt["n_classes"],
                _ef_ckpt.get("hidden", 256), _ef_ckpt.get("dropout", 0.3),
            ).to(_ef_dev)
            _ef_model.load_state_dict(_ef_ckpt["state_dict"])
            _ef_model.eval()

            _ef_emb_data   = _t.load(_ef_emb_path, map_location="cpu")
            _ef_train_embs = _ef_emb_data["train"].numpy().astype(np.float32)  # (n_train, 768)
            _ef_test_embs  = _ef_emb_data["test"].numpy().astype(np.float32)   # (n_test, 768)

            # Build text_str → embedding lookup from both splits
            _ef_text_to_emb: dict = {}
            for _i, _t_str in enumerate(dataset.X_train_text):
                _ef_text_to_emb[str(_t_str)] = _ef_train_embs[_i]
            for _i, _t_str in enumerate(dataset.X_test_text):
                _ef_text_to_emb.setdefault(str(_t_str), _ef_test_embs[_i])

            # Training-sample prediction cache
            _n_tr = len(dataset.X_train_static)
            _ef_cache: dict = {}
            _BS = 256
            for _bs in range(0, _n_tr, _BS):
                _be = min(_bs + _BS, _n_tr)
                _inp = _t.tensor(np.concatenate([
                    _ef_train_embs[_bs:_be],
                    dataset.X_train_static[_bs:_be].astype(np.float32),
                ], axis=1), dtype=_t.float32).to(_ef_dev)
                with _t.no_grad():
                    _preds = _ef_model(_inp).argmax(dim=-1).cpu().numpy()
                for _i, _gi in enumerate(range(_bs, _be)):
                    _ef_cache[(
                        dataset.X_train_static[_gi].astype(np.float32).tobytes(),
                        str(dataset.X_train_text[_gi]),
                    )] = float(_preds[_i])
            print(f"  [early fusion] Cached {len(_ef_cache):,} training-sample predictions.")

            def _predict_fn(x_tab, x_ts, text_candidate,
                            _model=_ef_model, _dev=_ef_dev,
                            _cache=_ef_cache, _lookup=_ef_text_to_emb,
                            _lock=_predict_fn_lock):
                import torch as _t2
                text = str(text_candidate) if text_candidate is not None else ""
                if x_tab is not None:
                    _hit = _cache.get((np.asarray(x_tab, dtype=np.float32).tobytes(), text))
                    if _hit is not None:
                        return _hit
                emb = _lookup.get(text)
                if emb is None:
                    return 0.0  # unknown text; return majority class
                tab = np.asarray(x_tab, dtype=np.float32)
                inp = _t2.tensor(np.concatenate([emb, tab])[None, :], dtype=_t2.float32).to(_dev)
                with _t2.no_grad():
                    return float(_model(inp).argmax(dim=-1).cpu().item())

            print(f"[early fusion] predict_fn ready using {_ef_path.name}")
        except Exception as _e:
            print(f"[warn] Early-fusion model failed to load: {_e}")
    else:
        print(f"[warn] Early-fusion files missing — predict_fn unavailable.")

# ---- Late fusion predict_fn (tfidf+logreg text + gbt tabular, averaged) -----
if _fusion_strategy == "late":
    import pickle as _pickle
    _lf_text_path = _DATA_DIR / "best_model_late_fusion_tfidf_logreg_text.pkl"
    _lf_tab_path  = _DATA_DIR / "best_model_late_fusion_gbt_tabular.pkl"
    if _lf_text_path.exists() and _lf_tab_path.exists():
        try:
            from sklearn.pipeline import Pipeline as _Pipeline
            with open(_lf_text_path, "rb") as _f:
                _lf_text_obj = _pickle.load(_f)
            if isinstance(_lf_text_obj, dict) and "tfidf" in _lf_text_obj:
                _lf_text_model = _Pipeline([("tfidf", _lf_text_obj["tfidf"]),
                                            ("clf",   _lf_text_obj["model"])])
            else:
                _lf_text_model = _lf_text_obj if not isinstance(_lf_text_obj, dict) \
                                 else _lf_text_obj["model"]
            with open(_lf_tab_path, "rb") as _f:
                _lf_tab_obj = _pickle.load(_f)
            _lf_tab_model = _lf_tab_obj if not isinstance(_lf_tab_obj, dict) \
                            else _lf_tab_obj["model"]

            # Training-sample prediction cache
            _n_tr = len(dataset.X_train_static)
            _lf_cache: dict = {}
            for _gi in range(_n_tr):
                _txt   = str(dataset.X_train_text[_gi])
                _tab   = dataset.X_train_static[_gi].astype(np.float32)
                _p_txt = _lf_text_model.predict_proba([_txt])[0]
                _p_tab = _lf_tab_model.predict_proba(_tab[None, :])[0]
                _avg   = (_p_txt + _p_tab) / 2.0
                _lf_cache[(dataset.X_train_static[_gi].astype(np.float32).tobytes(),
                           _txt)] = float(np.argmax(_avg))
            print(f"  [late fusion] Cached {len(_lf_cache):,} training-sample predictions.")

            def _predict_fn(x_tab, x_ts, text_candidate,
                            _cache=_lf_cache,
                            _tmod=_lf_text_model, _tabmod=_lf_tab_model):
                text = str(text_candidate) if text_candidate is not None else ""
                if x_tab is not None:
                    _hit = _cache.get((np.asarray(x_tab, dtype=np.float32).tobytes(), text))
                    if _hit is not None:
                        return _hit
                p_txt = _tmod.predict_proba([text])[0]
                p_tab = _tabmod.predict_proba(np.asarray(x_tab, dtype=np.float32)[None, :])[0]
                return float(np.argmax((p_txt + p_tab) / 2.0))

            print("[late fusion] predict_fn ready (tfidf_logreg_text + gbt_tabular averaged).")
        except Exception as _e:
            print(f"[warn] Late-fusion models failed to load: {_e}")
    else:
        print("[warn] Late-fusion model files missing — predict_fn unavailable.")

# — Italian fastText (optional, "custom" encoder slot) —
_italian_ft_embed_fn: Optional[object] = None
if args.italian_ft_path is not None and not args.no_italian_ft:
    ft_path = Path(args.italian_ft_path)
    if not ft_path.exists():
        raise FileNotFoundError(f"Italian fastText model not found: {ft_path}")
    print(f"Loading Italian fastText model from {ft_path} …")
    try:
        import fasttext
        _ft_model = fasttext.load_model(str(ft_path))
        _ft_dim   = _ft_model.get_dimension()

        def _italian_ft_embed_fn(texts, _ft=_ft_model):
            # Raw sentence vectors — pipeline normalises for cosine only.
            out = []
            for text in texts:
                out.append(_ft.get_sentence_vector(str(text).replace("\n", " ")).astype(np.float32))
            return np.stack(out)

    except ImportError:
        # Fallback: gensim's FastText loader (slower but no extra dependency)
        from gensim.models.fasttext import load_facebook_vectors
        _ft_kv  = load_facebook_vectors(str(ft_path))
        _ft_dim = _ft_kv.vector_size

        def _italian_ft_embed_fn(texts, _kv=_ft_kv):
            # Raw mean-pooled vectors — pipeline normalises for cosine only.
            out = []
            for text in texts:
                tokens = str(text).lower().split()
                vecs   = [_kv.get_vector(t) for t in tokens if t]
                emb    = np.mean(vecs, axis=0).astype(np.float32) if vecs \
                         else np.zeros(_kv.vector_size, dtype=np.float32)
                out.append(emb)
            return np.stack(out)

    # Expose to TextNN via the "custom" encoder slot
    text_bk["text_embed_fn"] = _italian_ft_embed_fn
    text_encoders.insert(-1, "custom")   # add before "raw"
    print(f"Italian fastText loaded: {_ft_dim}-dim")

# — word2vec (optional) —
_w2v_embed_fn: Optional[object] = None
if args.word2vec_path is not None:
    w2v_path = Path(args.word2vec_path)
    if not w2v_path.exists():
        raise FileNotFoundError(f"word2vec file not found: {w2v_path}")

    print(f"Loading word2vec from {w2v_path} …")
    from gensim.models import KeyedVectors
    _w2v_kv = (
        KeyedVectors.load(str(w2v_path))
        if str(w2v_path).endswith(".kv")
        else KeyedVectors.load_word2vec_format(str(w2v_path), binary=str(w2v_path).endswith(".bin"))
    )

    def _w2v_embed_fn(texts, _kv=_w2v_kv):
        # Raw mean-pooled vectors — pipeline normalises for cosine only.
        out = []
        for text in texts:
            tokens = str(text).lower().split()
            vecs = [_kv[t] for t in tokens if t in _kv]
            emb = np.mean(vecs, axis=0).astype(np.float32) if vecs \
                  else np.zeros(_kv.vector_size, dtype=np.float32)
            out.append(emb)
        return np.stack(out)

    text_bk["word2vec_model"] = _w2v_kv   # expose to TextNN generator
    text_encoders.insert(-1, "word2vec")   # add before "raw"
    print(f"word2vec loaded: {_w2v_kv.vector_size}-dim, {len(_w2v_kv):,} tokens")

# — bert (already loaded in text_bk when not --no-bert) —
_bert_embed_fn: Optional[object] = None
if not args.no_bert:
    from cf_lib.counterfactual_evaluation_helpers import _HF_BACKEND_LOCK, _make_embed_fn_from_e5_kwargs
    _bert_embed_fn = _make_embed_fn_from_e5_kwargs(
        tokenizer = text_bk["bert_tokenizer"],
        model     = text_bk["bert_model"],
        device    = text_bk["bert_device"],
    )
else:
    from cf_lib.counterfactual_evaluation_helpers import _HF_BACKEND_LOCK


def _precompute_text_embeddings_once(name: str, embed_fn) -> Dict[str, np.ndarray]:
    """Embed the full train+test corpus once and split it back into two arrays."""
    all_texts = train_texts + test_texts
    print(f"Precomputing {name} embeddings for {len(all_texts)} texts …")
    all_embs = np.asarray(embed_fn(all_texts), dtype=np.float32)
    n_train = len(train_texts)
    return {
        "train": all_embs[:n_train].copy(),
        "test": all_embs[n_train:].copy(),
    }


_precomputed_text_embeddings_by_encoder: Dict[str, Dict[str, np.ndarray]] = {
    "tfidf": _precompute_text_embeddings_once("tfidf", _tfidf_embed_fn),
}
if _bert_embed_fn is not None:
    _precomputed_text_embeddings_by_encoder["bert"] = _precompute_text_embeddings_once(
        "bert",
        _bert_embed_fn,
    )
if _italian_ft_embed_fn is not None:
    _precomputed_text_embeddings_by_encoder["custom"] = _precompute_text_embeddings_once(
        "custom",
        _italian_ft_embed_fn,
    )
if _w2v_embed_fn is not None:
    _precomputed_text_embeddings_by_encoder["word2vec"] = _precompute_text_embeddings_once(
        "word2vec",
        _w2v_embed_fn,
    )
text_bk["precomputed_text_embeddings_by_encoder"] = _precomputed_text_embeddings_by_encoder

# ---------------------------------------------------------------------------
# Helpers shared by objectives and multimodal generator factories
# ---------------------------------------------------------------------------
def _get_embed_fn_for_encoder(encoder: str):
    """Return the embed_fn that matches the search encoder.

    "raw" → tfidf fallback (no embedding space for direct metrics).
    """
    if encoder == "bert" and _bert_embed_fn is not None:
        return _bert_embed_fn
    if encoder == "custom" and _italian_ft_embed_fn is not None:
        return _italian_ft_embed_fn
    if encoder == "word2vec" and _w2v_embed_fn is not None:
        return _w2v_embed_fn
    return _tfidf_embed_fn  # tfidf or raw → tfidf fallback


def _get_precomputed_embeddings_for_encoder(encoder: str) -> Optional[Dict[str, np.ndarray]]:
    """Return cached train/test embeddings matched to the combo's text encoder."""
    if encoder == "bert":
        return _precomputed_text_embeddings_by_encoder.get("bert")
    if encoder == "custom":
        return _precomputed_text_embeddings_by_encoder.get("custom")
    if encoder == "word2vec":
        return _precomputed_text_embeddings_by_encoder.get("word2vec")
    # tfidf and raw both use tfidf embeddings for EarlyFusion space construction.
    return _precomputed_text_embeddings_by_encoder.get("tfidf")


def _metric_to_static_dist_fn(metric: str):
    """Convert a metric name to a pairwise-distance callable for static features."""
    if metric == "manhattan":
        return manhattan_distances
    if metric == "hamming":
        from sklearn.metrics import pairwise_distances as _pd
        return lambda A, B: _pd(A, B, metric="hamming")
    return euclidean_distances  # default / euclidean


# ---------------------------------------------------------------------------
# Per-combo objectives factory
# encoder used for search → same encoder for evaluation;
# "raw" search → tfidf for evaluation.
# The runner (run_distance_ablation) automatically wraps the returned embed_fn
# with a lookup-backed version using precomputed_text_embeddings_by_encoder,
# so no GPU embed calls occur during the parallel phase.
# ---------------------------------------------------------------------------
def _objectives_kwargs_factory(text_cfg, image_cfg):
    """Return compute_objectives kwargs matched to this combo's text encoder."""
    encoder = (text_cfg or {}).get("encoder", "raw")
    # "raw" has no embedding space → tfidf for objectives
    obj_encoder = "tfidf" if encoder == "raw" else encoder
    embed_fn = _get_embed_fn_for_encoder(obj_encoder)
    kwargs = {
        "text_objective_context": {"embed_fn": embed_fn},
        "y_target": joy_value,
        "proximity_normalizer": _prox_norm,
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
    if _predict_fn is not None:
        kwargs["predict_fn"] = _predict_fn
    return kwargs


# ---------------------------------------------------------------------------
# Per-combo multimodal generator factory
# Adds ModalityWisePrototypeSynthesis, MultimodalConsensusRetrieval, and EarlyFusionNN to each combo.
# ---------------------------------------------------------------------------
def _multimodal_generators_factory(
    tab_cfg,
    ts_cfg,
    text_cfg,
    text_backend_kwargs,
    image_cfg,
    image_backend_kwargs,
):
    """Build Frankenstein, Combined, and EarlyFusion generators for this combo."""
    encoder       = (text_cfg or {}).get("encoder", "raw")
    primary_tab   = dataset.primary_tabular_name
    tab_metric    = (tab_cfg  or {}).get(primary_tab, "euclidean")
    embed_fn      = _get_embed_fn_for_encoder(encoder)
    precomputed_text = _get_precomputed_embeddings_for_encoder(encoder)
    static_dist   = _metric_to_static_dist_fn(tab_metric)

    extras = {
        "MPS": ModalityWisePrototypeSynthesis(
            k=args.k,
            k_search=args.k_search_combined,
            static_dist_fn=static_dist,
            e5_embed_fn=embed_fn,
        ),
        "MC-R": MultimodalConsensusRetrieval(
            k=args.k,
            k_search=args.k_search_combined,
            static_dist_fn=static_dist,
            e5_embed_fn=embed_fn,
        ),
        "EarlyFusion": EarlyFusionNN(
            k=args.k,
            distance_metric=tab_metric,
            e5_embed_fn=embed_fn,
            precomputed_train_text_embeddings=(
                None if precomputed_text is None else precomputed_text["train"]
            ),
            precomputed_test_text_embeddings=(
                None if precomputed_text is None else precomputed_text["test"]
            ),
        ),
    }

    # IntermediateFusion: latent space is fixed (model-defined), tab metric used for distance
    if _torch_latents is not None:
        extras["IntermediateFusion"] = PyTorchLatentNN(
            k=args.k,
            train_latents=_torch_latents[0],
            test_latents=_torch_latents[1],
            distance_metric=tab_metric,
        )

    return extras


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------
_default_run_names = {
    "intermediate": "fusion_intermediate_deep_k50",
    "early":        "fusion_early_mlp_k50",
    "late":         "fusion_late_nondp_k50",
}
_run_name = args.run_name or _default_run_names[_fusion_strategy]

print(f"\nRunning distance ablation …")
print(f"  Fusion strategy : {_fusion_strategy}  →  {_run_name}")
print(f"  Samples : {len(sadness_indices)} (predicted '{label_classes[source_value]}')")
print(f"  Target  : {joy_value} ({label_classes[joy_value]})")
print(f"  k       : {args.k}")
print(f"  n_jobs  : {args.n_jobs}")
print(f"  Tab metrics   : {tab_metrics}")
print(f"  Text encoders : {text_encoders}")
print()

if args.eval_pkls:
    from evaluate_k_ablation import evaluate_k_ablation
    _eval_dirs = [Path(d.strip()) for d in args.eval_pkls.split(",")]
    _k_vals = [int(x) for x in args.k_values.split(",")]
    evaluate_k_ablation(
        run_dirs=_eval_dirs,
        dataset=dataset,
        objectives_kwargs_factory=_objectives_kwargs_factory,
        k_values=_k_vals,
        output_path=_eval_dirs[0] / f"k_ablation_summary_k{'_'.join(str(k) for k in _k_vals)}.json",
    )
else:
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
        output_dir                = args.output_dir,
        run_name                  = _run_name,
        save_full                 = args.save_full,
        max_combinations          = args.max_combinations,
        n_jobs                    = args.n_jobs,
        objectives_kwargs_factory = _objectives_kwargs_factory,
        extra_generators_factory  = _multimodal_generators_factory,
    )
