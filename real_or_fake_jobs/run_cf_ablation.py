"""Counterfactual ablation: fake → real on job postings.

Finds all test samples predicted as "fake", then sweeps distance metric
combinations via cf_lib's run_distance_ablation to generate counterfactuals
that flip each sample to "real".

Three separate FieldTextNN generators are run per combo — one for each text
field (description, company_profile, requirements) — so the ablation shows
which field drives the best counterfactuals.

MultiFieldFrankensteinNN runs 4 independent NN branches (tabular + desc +
profile + reqs) with no intersection.  Candidate i draws its static features
from the median of the tabular-NN top-i neighbours, its description from
desc-NN[i], its company_profile from profile-NN[i], and its requirements from
reqs-NN[i] (chimeric).  All three text fields are stored explicitly in every
candidate dict and passed directly to predict_fn — no training-set lookup is
required.

MultiFieldCombinedNN intersects all four independent candidate sets (tabular ∩
desc ∩ profile ∩ reqs) and selects a single consistent training sample,
ranked by summed distance.  No chimeric mixing.

EarlyFusionNN concatenates the three field embeddings along with the tabular
features and searches in that joint space.

All candidate dicts store "text" = description, "company_profile", and
"requirements" explicitly, making per-field proximity objectives and predict_fn
calls consistent across every generator.

Run
---
    cd real_or_fake_jobs
    python run_cf_ablation.py --gpu 7

    # Smoke test (3 samples, no bert)
    python run_cf_ablation.py --gpu 7 --max-samples 3 --no-bert

    # With word2vec
    python run_cf_ablation.py --gpu 7 --max-samples 3 --no-bert \\
        --word2vec-path data/glove_42B_300.kv

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
import threading
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Make cf_lib and examples importable from the real_or_fake_jobs subdirectory
_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_ROOT), str(_ROOT / "examples")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

from job_cf_factory import build_job_dataset
from run_distance_ablation import run_distance_ablation  # from examples/
from cf_lib.base import CounterfactualGenerator
from cf_lib.multimodal import CombinedNN, EarlyFusionNN
from cf_lib.unimodal import TabularNN, TextNN
from counterfactual_helpers import find_k_closest_latent, find_k_closest_static

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Fake→Real counterfactual ablation on job postings."
)
parser.add_argument("--gpu",               type=int, default=None)
parser.add_argument("--k",                 type=int, default=20,
                    help="Number of nearest neighbours per generator (default: 20).")
parser.add_argument("--max-samples",       type=int, default=None,
                    help="Cap on how many fake samples to process (default: all).")
parser.add_argument("--max-combinations",  type=int, default=None,
                    help="Cap on metric combinations to try (default: all).")
parser.add_argument("--n-jobs",            type=int, default=4,
                    help="Number of ablation combinations to run in parallel.")
parser.add_argument("--output-dir",        type=str, default="data/ablation_runs")
parser.add_argument("--run-name",          type=str, default=None)
parser.add_argument("--save-full",         action="store_true",
                    help="Save full candidate dicts as .pkl per combination.")
parser.add_argument("--no-bert",           action="store_true",
                    help="Skip loading the BERT text backend (faster for smoke tests).")
parser.add_argument("--word2vec-path",     type=str, default="data/word2vec_google_news_300.kv",
                    help="Path to a gensim word2vec .bin or .kv file.")
parser.add_argument("--source-class",      type=str, default="fake",
                    help="Class to explain (default: fake).")
parser.add_argument("--target-class",      type=str, default="real",
                    help="Counterfactual target class (default: real).")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load dataset and predictions
# ---------------------------------------------------------------------------
produced = build_job_dataset(gpu=args.gpu, load_bert=not args.no_bert)

dataset       = produced["dataset"]
text_bk       = produced["text_backend_kwargs"]
label_classes = produced["label_classes"]
y_pred        = produced["y_pred"]

if y_pred is None:
    raise FileNotFoundError(
        "data/y_pred.npy not found — run evaluate.py first."
    )

# Per-field text arrays (description is already in dataset.X_train_text)
X_train_description     = produced["X_train_description"]
X_test_description      = produced["X_test_description"]
X_train_company_profile = produced["X_train_company_profile"]
X_test_company_profile  = produced["X_test_company_profile"]
X_train_requirements    = produced["X_train_requirements"]
X_test_requirements     = produced["X_test_requirements"]

_FIELD_NAMES  = ["description", "company_profile", "requirements"]
_FIELD_TRAINS = [X_train_description, X_train_company_profile, X_train_requirements]
_FIELD_TESTS  = [X_test_description,  X_test_company_profile,  X_test_requirements]

# ---------------------------------------------------------------------------
# Resolve source / target class indices
# ---------------------------------------------------------------------------
lc_lower     = [c.lower() for c in label_classes]
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

fake_indices = [int(i) for i, p in enumerate(y_pred) if p == source_value]
if args.max_samples is not None:
    fake_indices = fake_indices[: args.max_samples]

print(f"\nLabel classes : {label_classes}")
print(f"source index  : {source_value} ({label_classes[source_value]})  |  "
      f"target index: {target_value} ({label_classes[target_value]})")
print(f"Test samples predicted as '{label_classes[source_value]}': {len(fake_indices)}")
if len(fake_indices) == 0:
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
# TF-IDF — fit on all three fields combined for shared vocabulary coverage
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

print("\nFitting TF-IDF on all three text fields …")
_all_train_texts = (
    [str(t) for t in X_train_description]
    + [str(t) for t in X_train_company_profile]
    + [str(t) for t in X_train_requirements]
)
_tfidf_vec = TfidfVectorizer(max_features=10_000, sublinear_tf=True)
_tfidf_vec.fit(_all_train_texts)


def _tfidf_embed_fn(texts):
    return _tfidf_vec.transform(texts).toarray().astype(np.float32)


# — tabular LOF (pre-fitted once; reused across all combos/candidates) —
# tabular_plausibility() fits a fresh LOF on the full training set per call,
# which takes ~1.5 s × 12 000 candidates/combo → hours of wall time.
# Pre-fitting once and passing via plausibility_normalizer replaces this with
# a cheap score_samples() lookup (read-only, thread-safe).
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


# ---------------------------------------------------------------------------
# BERT embed_fn (when not --no-bert)
# ---------------------------------------------------------------------------
_bert_embed_fn: Optional[object] = None
if not args.no_bert:
    from counterfactual_evaluation_helpers import _make_embed_fn_from_e5_kwargs
    _bert_embed_fn = _make_embed_fn_from_e5_kwargs(
        tokenizer = text_bk["bert_tokenizer"],
        model     = text_bk["bert_model"],
        device    = text_bk["bert_device"],
    )

# ---------------------------------------------------------------------------
# word2vec (optional)
# ---------------------------------------------------------------------------
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
        else KeyedVectors.load_word2vec_format(
            str(w2v_path), binary=str(w2v_path).endswith(".bin")
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
    text_encoders.insert(-1, "word2vec")
    print(f"word2vec loaded: {_w2v_kv.vector_size}-dim, {len(_w2v_kv):,} tokens")

# ---------------------------------------------------------------------------
# Precompute embeddings per field and per encoder, then concatenate
#
# _precomputed_by_field[field_name][encoder] = {"train": ndarray, "test": ndarray}
# _precomputed_concat[encoder]               = {"train": ndarray, "test": ndarray}
#   where the concat array is [desc_emb || profile_emb || reqs_emb] row-wise
# ---------------------------------------------------------------------------
def _precompute_field(name: str, embed_fn, train_arr, test_arr) -> Dict[str, np.ndarray]:
    all_texts = [str(t) for t in train_arr] + [str(t) for t in test_arr]
    print(f"Precomputing {name} embeddings for {len(all_texts)} texts …")
    all_embs = np.asarray(embed_fn(all_texts), dtype=np.float32)
    n_train  = len(train_arr)
    return {"train": all_embs[:n_train].copy(), "test": all_embs[n_train:].copy()}


_encoder_embed_fns = [("tfidf", _tfidf_embed_fn)]
if _bert_embed_fn is not None:
    _encoder_embed_fns.append(("bert", _bert_embed_fn))
if _w2v_embed_fn is not None:
    _encoder_embed_fns.append(("word2vec", _w2v_embed_fn))

_precomputed_by_field: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
    fname: {} for fname in _FIELD_NAMES
}
for enc_name, embed_fn in _encoder_embed_fns:
    for fname, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS):
        _precomputed_by_field[fname][enc_name] = _precompute_field(
            f"{fname}/{enc_name}", embed_fn, ftrain, ftest
        )

# Concatenated [desc || profile || reqs] embeddings per encoder
_precomputed_concat: Dict[str, Dict[str, np.ndarray]] = {}
for enc_name, _ in _encoder_embed_fns:
    _precomputed_concat[enc_name] = {
        split: np.concatenate(
            [_precomputed_by_field[fname][enc_name][split] for fname in _FIELD_NAMES],
            axis=1,
        )
        for split in ("train", "test")
    }

# Build per-field text-string → precomputed-vector lookup tables.
# Both train and test strings are indexed so embed_fn calls in objective
# evaluation hit the cache without any GPU inference during the parallel phase.
_field_emb_lookup: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {
    fname: {
        enc_name: {
            **{str(t): v
               for t, v in zip(ftrain, _precomputed_by_field[fname][enc_name]["train"])},
            **{str(t): v
               for t, v in zip(ftest,  _precomputed_by_field[fname][enc_name]["test"])},
        }
        for enc_name, _ in _encoder_embed_fns
    }
    for fname, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS)
}

# Expose description precomputed embeddings to the base TextNN via text_bk
text_bk["precomputed_text_embeddings_by_encoder"] = {
    fname: {
        enc_name: _precomputed_by_field[fname][enc_name]
        for enc_name, _ in _encoder_embed_fns
    }
    for fname in _FIELD_NAMES
}
text_bk["auto_text_branch_generators"] = False

# ---------------------------------------------------------------------------
# FieldTextNN — TextNN on a specific field; output "text" is always description
# so that objectives are evaluated on the same field across all generators.
# ---------------------------------------------------------------------------
class FieldTextNN(CounterfactualGenerator):
    """TextNN operating on one named text field.

    Searches for nearest neighbours in the given field's embedding space (or
    using direct metrics on that field's text).  Each output candidate's
    ``"text"`` / ``"text_input"`` is replaced with the selected training
    sample's *description*, keeping proximity objectives comparable across all
    three field generators.
    """

    def __init__(
        self,
        field_name: str,
        X_train_field,
        X_test_field,
        inner_texnn: TextNN,
        X_train_desc,
    ):
        self._field_name  = field_name
        self._X_train     = X_train_field
        self._X_test      = X_test_field
        self._inner       = inner_texnn
        self._train_desc  = np.asarray(X_train_desc, dtype=object).reshape(-1)

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None):
        candidates = self._inner.generate(dataset, sample_idx, target_value=target_value, k=k)
        # Replace "text" / "text_input" with description for consistent objectives
        for cand in candidates:
            src = cand.get("source_train_idx")
            if src is not None:
                cand["text"]       = str(self._train_desc[src])
                cand["text_input"] = self._train_desc[src]
                cand.setdefault("texts", {})["description"] = str(self._train_desc[src])
                cand.setdefault("text_inputs", {})["description"] = self._train_desc[src]
        return candidates

    def generate_batch(self, dataset, sample_indices, model=None, target_value=0, k=None):
        batch = self._inner.generate_batch(
            dataset,
            sample_indices,
            model=model,
            target_value=target_value,
            k=k,
        )
        for candidates in batch.values():
            for cand in candidates:
                src = cand.get("source_train_idx")
                if src is not None:
                    cand["text"] = str(self._train_desc[src])
                    cand["text_input"] = self._train_desc[src]
                    cand.setdefault("texts", {})["description"] = str(self._train_desc[src])
                    cand.setdefault("text_inputs", {})["description"] = self._train_desc[src]
        return batch


# ---------------------------------------------------------------------------
# _run_embedding_nn_search — NN in a pre-computed embedding space
# ---------------------------------------------------------------------------
def _run_embedding_nn_search(
    sample_idx:   int,
    emb_train:    np.ndarray,
    emb_test:     np.ndarray,
    y_train:      np.ndarray,
    target_value: int,
    k_search:     int,
    metric:       str = "cosine",
):
    """Find k_search nearest neighbours in a pre-computed embedding space.

    Returns (idx_dict, dist_dict) keyed by sample_idx, compatible with the
    precomputed interface of FrankensteinNN / CombinedNN.
    """
    candidate_idx = np.flatnonzero(np.asarray(y_train) == target_value)
    query         = emb_test[[sample_idx]]
    cands         = emb_train[candidate_idx]
    dists_row     = pairwise_distances(query, cands, metric=metric)[0]
    k_eff         = min(k_search, len(candidate_idx))
    part          = np.argpartition(dists_row, k_eff - 1)[:k_eff]
    order         = part[np.argsort(dists_row[part])]
    top_train_idx = candidate_idx[order]
    top_dists     = dists_row[order]
    return {sample_idx: top_train_idx}, {sample_idx: top_dists}


def _run_embedding_nn_search_batch(
    sample_indices,
    emb_train:    np.ndarray,
    emb_test:     np.ndarray,
    y_train:      np.ndarray,
    target_value: int,
    k_search:     int,
    metric:       str = "cosine",
):
    """Find k_search nearest neighbours in a pre-computed embedding space."""
    selected = np.asarray([int(idx) for idx in sample_indices], dtype=int)
    candidate_idx = np.flatnonzero(np.asarray(y_train) == target_value)
    if candidate_idx.size == 0:
        empty_idx = np.empty((0,), dtype=int)
        empty_dist = np.empty((0,), dtype=float)
        return (
            {int(idx): empty_idx for idx in selected},
            {int(idx): empty_dist for idx in selected},
        )

    queries = emb_test[selected]
    cands = emb_train[candidate_idx]
    dist_matrix = pairwise_distances(queries, cands, metric=metric)
    k_eff = min(k_search, len(candidate_idx))

    idx_dict = {}
    dist_dict = {}
    for r, sample_idx in enumerate(selected):
        dists_row = dist_matrix[r]
        if k_eff == len(candidate_idx):
            order = np.argsort(dists_row)[:k_eff]
        else:
            part = np.argpartition(dists_row, k_eff - 1)[:k_eff]
            order = part[np.argsort(dists_row[part])]
        idx_dict[int(sample_idx)] = candidate_idx[order]
        dist_dict[int(sample_idx)] = dists_row[order]
    return idx_dict, dist_dict


# ---------------------------------------------------------------------------
# MultiFieldFrankensteinNN — 4 independent branches: tabular + desc + profile + reqs
# ---------------------------------------------------------------------------
class MultiFieldFrankensteinNN(CounterfactualGenerator):
    """Frankenstein with 4 independent NN branches: tabular + desc + profile + reqs.

    Static component : median of tabular-NN[1..i] per candidate (synthetic,
                       independent from the text component).
    Text component   : each field drawn from its own NN independently:
                         - description    from desc-NN[i]
                         - company_profile from profile-NN[i]
                         - requirements   from reqs-NN[i]

    All three text fields are stored in every candidate:
    ``candidate["text"]`` / ``candidate["text_input"]`` = description,
    ``candidate["company_profile"]`` = company profile from profile-NN[i],
    ``candidate["requirements"]`` = requirements from reqs-NN[i].

    ``predict_fn`` and per-field text objectives (via ``_text_modalities_fn``)
    use all three fields directly from the candidate, so no lookup approximation
    is needed.
    """

    def __init__(
        self,
        pre_desc:      Dict[str, np.ndarray],
        pre_profile:   Dict[str, np.ndarray],
        pre_reqs:      Dict[str, np.ndarray],
        y_train:       np.ndarray,
        vec_metric:    str = "cosine",
        k:             int = 5,
        k_search:      int = 50,
        static_dist_fn = None,
    ):
        from sklearn.metrics.pairwise import euclidean_distances as _euc
        self.k              = k
        self.k_search       = k_search
        self.static_dist_fn = static_dist_fn or _euc
        self._pre_desc      = pre_desc
        self._pre_profile   = pre_profile
        self._pre_reqs      = pre_reqs
        self._y_train       = np.asarray(y_train)
        self._vec_metric    = vec_metric

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None):
        return self.generate_batch(
            dataset,
            [sample_idx],
            model=model,
            target_value=target_value,
            k=k,
        ).get(int(sample_idx), [])

    def generate_batch(self, dataset, sample_indices, model=None, target_value=0, k=None):
        k = k if k is not None else self.k
        selected = [int(idx) for idx in sample_indices]

        # --- tabular branch (static features) ---
        idx_tab, _ = find_k_closest_static(
            X_train_static        = dataset.X_train_static,
            y_train               = dataset.y_train,
            X_test_static         = dataset.X_test_static,
            selected_test_indices = selected,
            target_value          = target_value,
            k                     = self.k_search,
            distance_fn           = self.static_dist_fn,
            return_indices        = True,
        )

        # --- 3 independent text-field branches ---
        desc_idx_d, _  = _run_embedding_nn_search_batch(
            selected, self._pre_desc["train"],    self._pre_desc["test"],
            self._y_train, target_value, self.k_search, self._vec_metric,
        )
        prof_idx_d, _  = _run_embedding_nn_search_batch(
            selected, self._pre_profile["train"], self._pre_profile["test"],
            self._y_train, target_value, self.k_search, self._vec_metric,
        )
        reqs_idx_d, _  = _run_embedding_nn_search_batch(
            selected, self._pre_reqs["train"],    self._pre_reqs["test"],
            self._y_train, target_value, self.k_search, self._vec_metric,
        )

        X_static  = np.asarray(dataset.X_train_static,    dtype=float)
        X_desc    = np.asarray(X_train_description,        dtype=object).reshape(-1)
        X_profile = np.asarray(X_train_company_profile,    dtype=object).reshape(-1)
        X_reqs    = np.asarray(X_train_requirements,       dtype=object).reshape(-1)

        batch_results = {}
        for sample_idx in selected:
            tab_idxs = np.asarray(idx_tab.get(int(sample_idx), []), dtype=int)
            desc_idxs = np.asarray(desc_idx_d.get(int(sample_idx), []), dtype=int)
            profile_idxs = np.asarray(prof_idx_d.get(int(sample_idx), []), dtype=int)
            reqs_idxs = np.asarray(reqs_idx_d.get(int(sample_idx), []), dtype=int)

            k_eff = min(k, len(tab_idxs), len(desc_idxs), len(profile_idxs), len(reqs_idxs))
            results = []
            for i in range(1, k_eff + 1):
                use_n = i
                tab_sub = tab_idxs[:use_n]
                static_med = (
                    X_static[tab_sub[0]].copy()
                    if use_n == 1
                    else np.median(X_static[tab_sub], axis=0)
                ).astype(float)

                desc_anc = int(desc_idxs[i - 1])
                profile_anc = int(profile_idxs[i - 1])
                reqs_anc = int(reqs_idxs[i - 1])

                desc_str = str(X_desc[desc_anc])
                profile_str = str(X_profile[profile_anc])
                reqs_str = str(X_reqs[reqs_anc])

                results.append({
                    "static":      static_med,
                    "ts":          {},
                    "tab":         {},
                    "text":        desc_str,
                    "text_input":  desc_str,
                    "image":       None,
                    "image_input": None,
                    "source_train_idx": desc_anc,
                    "source_indices": {
                        "static":       int(tab_idxs[i - 1]),
                        "text_desc":    desc_anc,
                        "text_profile": profile_anc,
                        "text_reqs":    reqs_anc,
                    },
                    "n_neighbors_used": use_n,
                    "distance_metric":  f"chimeric/{self._vec_metric}",
                    "company_profile": profile_str,
                    "requirements":    reqs_str,
                })
            batch_results[int(sample_idx)] = results
        return batch_results


# ---------------------------------------------------------------------------
# MultiFieldCombinedNN — intersection of 4 independent searches
# ---------------------------------------------------------------------------
class MultiFieldCombinedNN(CombinedNN):
    """CombinedNN with 4 independent searches: tabular + desc + profile + reqs.

    Intersects all four candidate sets (tabular ∩ desc-text ∩ profile-text ∩
    reqs-text) and ranks the common training samples by summed distance across
    all four spaces.  Falls back to empty candidates when the intersection is
    empty.

    Because all four modalities must agree on the *same* training sample, the
    selected sample's static features, description, company_profile, and
    requirements are all consistent — no chimeric mixing.  company_profile and
    requirements are annotated onto each candidate dict after selection so that
    predict_fn and per-field objectives receive all three text fields directly.
    """

    def __init__(
        self,
        pre_desc:    Dict[str, np.ndarray],
        pre_profile: Dict[str, np.ndarray],
        pre_reqs:    Dict[str, np.ndarray],
        y_train:     np.ndarray,
        vec_metric:  str = "cosine",
        **combined_kwargs,
    ):
        super().__init__(**combined_kwargs)
        self._pre_desc    = pre_desc
        self._pre_profile = pre_profile
        self._pre_reqs    = pre_reqs
        self._y_train     = np.asarray(y_train)
        self._vec_metric  = vec_metric

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None, precomputed=None):
        return self.generate_batch(
            dataset,
            [sample_idx],
            model=model,
            target_value=target_value,
            k=k,
            precomputed=precomputed,
        ).get(int(sample_idx), [])

    def generate_batch(self, dataset, sample_indices, model=None, target_value=0, k=None, precomputed=None):
        k  = k if k is not None else self.k
        selected = [int(idx) for idx in sample_indices]
        pc = dict(precomputed or {})
        avail = dataset.available_modalities

        active_idx_dist: dict = {}

        # --- tabular search ---
        if "tabular" in avail:
            if "tabular" in pc:
                active_idx_dist["tabular"] = pc["tabular"]
            else:
                idx_t, dist_t = find_k_closest_static(
                    X_train_static        = dataset.X_train_static,
                    y_train               = dataset.y_train,
                    X_test_static         = dataset.X_test_static,
                    selected_test_indices = selected,
                    target_value          = target_value,
                    k                     = self.k_search,
                    distance_fn           = self.static_dist_fn,
                    return_indices        = True,
                )
                active_idx_dist["tabular"] = (idx_t, dist_t)

        # --- 3 independent text-field searches ---
        for fname, pre in [
            ("text_desc",    self._pre_desc),
            ("text_profile", self._pre_profile),
            ("text_reqs",    self._pre_reqs),
        ]:
            idx_d, dist_d = _run_embedding_nn_search_batch(
                selected, pre["train"], pre["test"],
                self._y_train, target_value, self.k_search, self._vec_metric,
            )
            active_idx_dist[fname] = (idx_d, dist_d)

        if not active_idx_dist:
            return {int(idx): [] for idx in selected}

        X_profile = np.asarray(X_train_company_profile, dtype=object).reshape(-1)
        X_reqs    = np.asarray(X_train_requirements,    dtype=object).reshape(-1)
        batch_results = {}
        for sample_idx in selected:
            results = self._combined_partial(active_idx_dist, sample_idx, k, dataset)
            for cand in results:
                anchor = cand.get("source_train_idx")
                if anchor is not None:
                    cand["company_profile"] = str(X_profile[anchor])
                    cand["requirements"]    = str(X_reqs[anchor])
            batch_results[int(sample_idx)] = results
        return batch_results


# ---------------------------------------------------------------------------
# PyTorch model latent space (for IntermediateFusion generator)
# ---------------------------------------------------------------------------
class PyTorchLatentNN(CounterfactualGenerator):
    """Nearest-neighbour search in the DistilBERT classifier's penultimate-layer space."""

    def __init__(self, k: int, train_latents: np.ndarray, test_latents: np.ndarray,
                 distance_metric: str = "euclidean"):
        self.k               = k
        self.train_latents   = train_latents
        self.test_latents    = test_latents
        self.distance_metric = distance_metric

    def generate(self, dataset, sample_idx, model=None, target_value=0, k=None):
        return self.generate_batch(
            dataset,
            [sample_idx],
            model=model,
            target_value=target_value,
            k=k,
        ).get(int(sample_idx), [])

    def generate_batch(self, dataset, sample_indices, model=None, target_value=0, k=None):
        k = k if k is not None else self.k
        indices, _ = find_k_closest_latent(
            X_train_latent=self.train_latents,
            y_train=dataset.y_train,
            X_test_latent=self.test_latents,
            selected_test_indices=[int(idx) for idx in sample_indices],
            target_value=target_value,
            k=k,
            distance_metric=self.distance_metric,
        )
        return {
            int(idx): TabularNN._materialize(
                indices,
                int(idx),
                dataset,
                distance_metric_label=self.distance_metric or "euclidean",
            )
            for idx in sample_indices
        }


def _compute_torch_latents(torch_model, tokenizer, device, batch_size=16):
    """Extract penultimate-layer representations from the DistilBERT classifier.

    Hooks on model.classifier to capture the concatenated
    [desc_emb || profile_emb || reqs_emb || tab_emb] vector (3×text_hidden_dim + tab_out_dim).

    Returns (train_latents, test_latents) as float32 numpy arrays.
    """
    import torch as _t

    torch_model.eval()
    _buf: dict = {}

    def _hook(module, inp, out):
        _buf["z"] = inp[0].detach().cpu().numpy()

    handle = torch_model.classifier.register_forward_hook(_hook)

    def _run(X_static, X_desc, X_profile, X_reqs):
        n = len(X_static)
        rows = []
        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            descs    = [str(t) for t in X_desc[start:end]]
            profiles = [str(t) for t in X_profile[start:end]]
            reqs_    = [str(t) for t in X_reqs[start:end]]

            enc_d = tokenizer(descs,    max_length=512, padding="max_length",
                              truncation=True, return_tensors="pt")
            enc_p = tokenizer(profiles, max_length=512, padding="max_length",
                              truncation=True, return_tensors="pt")
            enc_r = tokenizer(reqs_,    max_length=512, padding="max_length",
                              truncation=True, return_tensors="pt")
            tab = _t.tensor(X_static[start:end], dtype=_t.float32).to(device)

            with _t.no_grad():
                torch_model(
                    enc_d["input_ids"].to(device), enc_d["attention_mask"].to(device),
                    enc_p["input_ids"].to(device), enc_p["attention_mask"].to(device),
                    enc_r["input_ids"].to(device), enc_r["attention_mask"].to(device),
                    tab,
                )
            rows.append(_buf["z"].copy())
        return np.vstack(rows).astype(np.float32)

    print("Extracting model latents for train split …")
    train_latents = _run(
        dataset.X_train_static,
        X_train_description, X_train_company_profile, X_train_requirements,
    )
    print("Extracting model latents for test split …")
    test_latents = _run(
        dataset.X_test_static,
        X_test_description, X_test_company_profile, X_test_requirements,
    )
    handle.remove()
    print(f"  Latent shape: {train_latents.shape[1]}-dim")
    return train_latents, test_latents


# ---------------------------------------------------------------------------
# Load PyTorch classifier and precompute latents
# ---------------------------------------------------------------------------
_torch_latents: Optional[tuple] = None
_predict_fn:    Optional[object] = None
_predict_fn_lock = threading.Lock()  # serializes live GPU inference (cache-miss path)
_DATA_DIR = Path(__file__).parent / "data"
_pt_path  = _DATA_DIR / "best_model.pt"

# Build a description → (desc, profile, reqs) lookup used as a fallback when
# source_train_idx is unavailable.  Does NOT depend on the model checkpoint.
_desc_to_fields: dict = {}
for _d, _p, _r in zip(
    X_train_description, X_train_company_profile, X_train_requirements
):
    _desc_to_fields.setdefault(str(_d), (str(_d), str(_p), str(_r)))

if _pt_path.exists():
    try:
        import torch
        import torch.nn as _nn
        from transformers import AutoModel as _AutoModel

        _TEXT_MODEL = "distilbert-base-uncased"

        class _TabHead(_nn.Module):
            def __init__(self, d_in, hidden_dims, dropout):
                super().__init__()
                layers, in_dim = [], d_in
                for h in hidden_dims:
                    layers += [_nn.Linear(in_dim, h), _nn.ReLU(), _nn.Dropout(dropout)]
                    in_dim = h
                self.net     = _nn.Sequential(*layers)
                self.out_dim = in_dim
            def forward(self, x): return self.net(x)

        class _MultimodalClassifier(_nn.Module):
            def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
                super().__init__()
                self.text_encoder = _AutoModel.from_pretrained(_TEXT_MODEL)
                enc_dim = self.text_encoder.config.hidden_size
                def _proj():
                    return _nn.Sequential(
                        _nn.Linear(enc_dim, text_hidden_dim), _nn.ReLU(), _nn.Dropout(dropout),
                    )
                self.proj_description     = _proj()
                self.proj_company_profile = _proj()
                self.proj_requirements    = _proj()
                self.tab_head             = _TabHead(d_tab, tab_hidden_dims, dropout)
                self.classifier           = _nn.Linear(
                    3 * text_hidden_dim + self.tab_head.out_dim, n_classes
                )

            def forward(self, desc_ids, desc_mask, profile_ids, profile_mask,
                        reqs_ids, reqs_mask, X_static):
                B       = desc_ids.size(0)
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

        _latent_tokenizer = text_bk.get("bert_tokenizer") or __import__(
            "transformers", fromlist=["AutoTokenizer"]
        ).AutoTokenizer.from_pretrained(_TEXT_MODEL)

        _torch_latents = _compute_torch_latents(
            _torch_model, _latent_tokenizer, _device_latent
        )

        # Precompute model predictions for all training samples in one batched pass.
        # Unimodal / Combined / EarlyFusion / IntermediateFusion candidates are pure
        # training samples — exact cache hits.  Only MultiFieldFrankensteinNN assembles
        # hybrids (tabular median + texts from three independent branches), so those
        # miss the cache and fall through to serialized live inference.
        print("Precomputing model predictions for all training samples …")
        _n_tr = len(dataset.X_train_static)
        _pred_rows = []
        for _bs in range(0, _n_tr, 16):
            _be = min(_bs + 16, _n_tr)
            _descs    = [str(t) for t in X_train_description[_bs:_be]]
            _profiles = [str(t) for t in X_train_company_profile[_bs:_be]]
            _reqs_    = [str(t) for t in X_train_requirements[_bs:_be]]
            _enc_d = _latent_tokenizer(_descs,    max_length=512, padding="max_length",
                                       truncation=True, return_tensors="pt")
            _enc_p = _latent_tokenizer(_profiles, max_length=512, padding="max_length",
                                       truncation=True, return_tensors="pt")
            _enc_r = _latent_tokenizer(_reqs_,    max_length=512, padding="max_length",
                                       truncation=True, return_tensors="pt")
            _ptab = torch.tensor(
                dataset.X_train_static[_bs:_be].astype(np.float32), dtype=torch.float32,
            ).to(_device_latent)
            with torch.no_grad():
                _plogits = _torch_model(
                    _enc_d["input_ids"].to(_device_latent), _enc_d["attention_mask"].to(_device_latent),
                    _enc_p["input_ids"].to(_device_latent), _enc_p["attention_mask"].to(_device_latent),
                    _enc_r["input_ids"].to(_device_latent), _enc_r["attention_mask"].to(_device_latent),
                    _ptab,
                )
            _pred_rows.append(_plogits.argmax(dim=-1).cpu().numpy().astype(np.float32))
        _train_pred_arr = np.concatenate(_pred_rows)
        _train_pred_cache: dict = {
            (
                dataset.X_train_static[_i].astype(np.float32).tobytes(),
                str(X_train_description[_i]),
                str(X_train_company_profile[_i]),
                str(X_train_requirements[_i]),
            ): float(_train_pred_arr[_i])
            for _i in range(_n_tr)
        }
        print(f"  Cached {len(_train_pred_cache):,} training-sample predictions.")

        def _predict_fn(x_tab_or_dict, x_ts, text_candidate_or_dict,
                        _model=_torch_model,
                        _tok=_latent_tokenizer,
                        _dev=_device_latent,
                        _cache=_train_pred_cache,
                        _lock=_predict_fn_lock):
            import torch as _t

            # Multi-instance path: called when text_modalities has > 1 entry
            # (i.e. when _text_modalities_fn is active and returns 3 text fields).
            # x_tab_or_dict is {"__primary__": array}, text_candidate_or_dict is
            # {"description": ..., "company_profile": ..., "requirements": ...}.
            if isinstance(text_candidate_or_dict, dict):
                desc    = str(text_candidate_or_dict.get("description",    "") or "")
                profile = str(text_candidate_or_dict.get("company_profile", "") or "")
                reqs    = str(text_candidate_or_dict.get("requirements",   "") or "")
                x_tab   = (x_tab_or_dict.get("__primary__")
                           if isinstance(x_tab_or_dict, dict) else x_tab_or_dict)
            else:
                # Single-instance fallback: look up profile/reqs from description.
                text    = str(text_candidate_or_dict) if text_candidate_or_dict is not None else ""
                desc, profile, reqs = _desc_to_fields.get(text, (text, "", ""))
                x_tab   = x_tab_or_dict

            # Fast path: cache hit (pure training-sample candidates).
            if x_tab is not None:
                _key = (np.asarray(x_tab, dtype=np.float32).tobytes(), desc, profile, reqs)
                _hit = _cache.get(_key)
                if _hit is not None:
                    return _hit

            # Cache miss: Frankenstein hybrid — serialized live inference.
            def _enc(t):
                return _tok([t], max_length=512, padding="max_length",
                            truncation=True, return_tensors="pt")

            tab = _t.tensor(np.asarray(x_tab, dtype=np.float32)[None, :],
                            dtype=_t.float32).to(_dev)
            with _lock:
                enc_d = _enc(desc)
                enc_p = _enc(profile)
                enc_r = _enc(reqs)
                with _t.no_grad():
                    logits = _model(
                        enc_d["input_ids"].to(_dev), enc_d["attention_mask"].to(_dev),
                        enc_p["input_ids"].to(_dev), enc_p["attention_mask"].to(_dev),
                        enc_r["input_ids"].to(_dev), enc_r["attention_mask"].to(_dev),
                        tab,
                    )
            return float(logits.argmax(dim=-1).cpu().item())

    except Exception as _e:
        print(f"[warn] Could not compute model latents — IntermediateFusion will be skipped: {_e}")
else:
    print(f"[warn] best_model.pt not found at {_pt_path} — IntermediateFusion will be skipped.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_embed_fn_for_encoder(encoder: str):
    """Return embed_fn for the given encoder; 'raw' falls back to tfidf."""
    if encoder == "bert" and _bert_embed_fn is not None:
        return _bert_embed_fn
    if encoder == "word2vec" and _w2v_embed_fn is not None:
        return _w2v_embed_fn
    return _tfidf_embed_fn


def _make_field_lookup_embed_fn(field_name: str, encoder: str):
    """Return an embed_fn for *field_name* backed by ``_field_emb_lookup``.

    All training and test texts for every field are precomputed, so the live
    fallback path is never exercised during normal runs.
    """
    lookup  = _field_emb_lookup.get(field_name, {}).get(encoder, {})
    live_fn = _get_embed_fn_for_encoder(encoder)

    def _fn(texts, _lk=lookup, _live=live_fn):
        vecs = [None] * len(texts)
        miss_pos, miss_txt = [], []
        for i, t in enumerate(texts):
            v = _lk.get(str(t) if t is not None else "")
            if v is not None:
                vecs[i] = v
            else:
                miss_pos.append(i)
                miss_txt.append(str(t) if t is not None else "")
        if miss_txt:
            live_embs = np.asarray(_live(miss_txt), dtype=np.float32)
            for j, pos in enumerate(miss_pos):
                vecs[pos] = live_embs[j]
        return np.stack(vecs)

    return _fn


def _get_field_precomputed(field_name: str, encoder: str) -> Optional[Dict[str, np.ndarray]]:
    """Return cached {train, test} embeddings for the given field and encoder."""
    return (
        _precomputed_by_field[field_name].get(encoder)
        or _precomputed_by_field[field_name].get("tfidf")
    )


def _get_concat_precomputed(encoder: str) -> Optional[Dict[str, np.ndarray]]:
    """Return cached concatenated 3-field embeddings for the given encoder."""
    return _precomputed_concat.get(encoder) or _precomputed_concat.get("tfidf")


def _metric_to_static_dist_fn(metric: str):
    if metric == "manhattan":
        return manhattan_distances
    if metric == "hamming":
        from sklearn.metrics import pairwise_distances as _pd
        return lambda A, B: _pd(A, B, metric="hamming")
    return euclidean_distances


# ---------------------------------------------------------------------------
# Per-combo objectives factory
# ---------------------------------------------------------------------------
def _objectives_kwargs_factory(text_cfg, image_cfg):
    encoder     = (text_cfg or {}).get("encoder", "raw")
    obj_encoder = "tfidf" if encoder == "raw" else encoder
    desc_ctx    = {"embed_fn": _make_field_lookup_embed_fn("description",    obj_encoder)}
    profile_ctx = {"embed_fn": _make_field_lookup_embed_fn("company_profile", obj_encoder)}
    reqs_ctx    = {"embed_fn": _make_field_lookup_embed_fn("requirements",   obj_encoder)}

    # Cache per-field test/train arrays as numpy object arrays for fast indexing.
    _Xtest_desc    = np.asarray(X_test_description,     dtype=object).reshape(-1)
    _Xtest_profile = np.asarray(X_test_company_profile, dtype=object).reshape(-1)
    _Xtest_reqs    = np.asarray(X_test_requirements,    dtype=object).reshape(-1)
    _Xtrain_prof   = np.asarray(X_train_company_profile, dtype=object).reshape(-1)
    _Xtrain_reqs   = np.asarray(X_train_requirements,   dtype=object).reshape(-1)

    def _text_modalities_fn(sample_idx, cand, text_factual):
        """Return text_modalities covering all 3 text fields for this candidate.

        Chimeric Frankenstein candidates store company_profile and requirements
        explicitly in the candidate dict (independently selected per-branch).
        All other generators select a single training sample, so profile/reqs
        are looked up from source_train_idx.
        """
        cand_desc = str(cand.get("text") or "")

        if "company_profile" in cand:
            # Explicitly stored (Frankenstein chimeric or Combined annotated)
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

    kwargs: dict = {
        "y_target": target_value,
        "_text_modalities_fn": _text_modalities_fn,
        "plausibility_normalizer": {
            "tab_lof":  _tab_lof,
            "tab_low":  _tab_lof_low,
            "tab_high": _tab_lof_high,
        },
    }
    if _predict_fn is not None:
        kwargs["predict_fn"] = _predict_fn
    return kwargs


# ---------------------------------------------------------------------------
# Per-combo generator factory
# ---------------------------------------------------------------------------
_k_search = min(50, args.k * 5)


def _multimodal_generators_factory(
    tab_cfg,
    ts_cfg,
    text_cfg,
    text_backend_kwargs,
    image_cfg,
    image_backend_kwargs,
):
    encoder    = (text_cfg or {}).get("encoder", "raw")
    metric     = (text_cfg or {}).get("metric", "cosine")
    primary_tab = dataset.primary_tabular_name
    tab_metric = (tab_cfg  or {}).get(primary_tab, "euclidean")

    static_dist = _metric_to_static_dist_fn(tab_metric)

    # Vector metric for concat-space searches (direct metrics have no embedding)
    vec_metric  = metric if encoder != "raw" and metric in (
        "cosine", "euclidean", "manhattan"
    ) else "cosine"

    # Precomputed per-field embeddings for this combo
    pre_desc    = _get_field_precomputed("description",     encoder)
    pre_profile = _get_field_precomputed("company_profile", encoder)
    pre_reqs    = _get_field_precomputed("requirements",    encoder)
    # Concatenated [desc || profile || reqs] for EarlyFusion
    pre_concat  = _get_concat_precomputed(encoder)

    # ---- 3 × FieldTextNN (one per text field) ----
    extras: dict = {}
    for fname, ftrain, ftest in zip(_FIELD_NAMES, _FIELD_TRAINS, _FIELD_TESTS):
        pre = _get_field_precomputed(fname, encoder)
        inner = TextNN(
            text_name                          = fname,
            k                                  = args.k,
            text_encoder                       = encoder,
            text_distance_metric               = metric,
            bert_tokenizer                     = text_backend_kwargs.get("bert_tokenizer"),
            bert_model                         = text_backend_kwargs.get("bert_model"),
            bert_device                        = text_backend_kwargs.get("bert_device"),
            word2vec_model                     = text_backend_kwargs.get("word2vec_model"),
            tfidf_vectorizer                   = _tfidf_vec,
            precomputed_train_text_embeddings  = pre["train"] if pre else None,
            precomputed_test_text_embeddings   = pre["test"]  if pre else None,
        )
        extras[f"Text_{fname}"] = FieldTextNN(fname, ftrain, ftest, inner, X_train_description)

    # ---- Frankenstein: 4 independent branches (tabular + desc + profile + reqs) ----
    # Static from tabular-NN[i]; each text field from its own NN[i] independently.
    if pre_desc is not None and pre_profile is not None and pre_reqs is not None:
        extras["Frankenstein"] = MultiFieldFrankensteinNN(
            pre_desc    = pre_desc,
            pre_profile = pre_profile,
            pre_reqs    = pre_reqs,
            y_train     = dataset.y_train,
            vec_metric  = vec_metric,
            k           = args.k,
            k_search    = _k_search,
            static_dist_fn = static_dist,
        )

    # ---- Combined: intersection of 4 sets (tabular ∩ desc ∩ profile ∩ reqs) ----
    if pre_desc is not None and pre_profile is not None and pre_reqs is not None:
        extras["Combined"] = MultiFieldCombinedNN(
            pre_desc    = pre_desc,
            pre_profile = pre_profile,
            pre_reqs    = pre_reqs,
            y_train     = dataset.y_train,
            vec_metric  = vec_metric,
            k           = args.k,
            k_search    = _k_search,
            static_dist_fn = static_dist,
        )

    # ---- EarlyFusion with concatenated 3-field text embeddings ----
    # For "raw" encoder there is no embedding so text is skipped by EarlyFusion.
    if pre_concat is not None:
        extras["EarlyFusion"] = EarlyFusionNN(
            k                                 = args.k,
            distance_metric                   = tab_metric,
            precomputed_train_text_embeddings = pre_concat["train"],
            precomputed_test_text_embeddings  = pre_concat["test"],
        )

    # ---- IntermediateFusion (model latent space) ----
    if _torch_latents is not None:
        extras["IntermediateFusion"] = PyTorchLatentNN(
            k                = args.k,
            train_latents    = _torch_latents[0],
            test_latents     = _torch_latents[1],
            distance_metric  = tab_metric,
        )

    return extras


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------
print(f"\nRunning distance ablation …")
print(f"  Samples : {len(fake_indices)} (predicted '{label_classes[source_value]}')")
print(f"  Target  : {target_value} ({label_classes[target_value]})")
print(f"  k       : {args.k}")
print(f"  n_jobs  : {args.n_jobs}")
print(f"  Tab metrics   : {tab_metrics}")
print(f"  Text encoders : {text_encoders}")
print(f"  Fields        : {_FIELD_NAMES}")
print()

run_distance_ablation(
    dataset                   = dataset,
    model                     = None,
    sample_indices            = fake_indices,
    target_value              = target_value,
    k                         = args.k,
    tab_metrics               = tab_metrics,
    ts_metrics                = [],
    text_encoders             = text_encoders,
    text_vector_metrics       = text_vector_metrics,
    text_direct_metrics       = text_direct_metrics,
    text_backend_kwargs       = text_bk,
    output_dir                = args.output_dir,
    run_name                  = args.run_name,
    save_full                 = args.save_full,
    max_combinations          = args.max_combinations,
    n_jobs                    = args.n_jobs,
    objectives_kwargs_factory = _objectives_kwargs_factory,
    extra_generators_factory  = _multimodal_generators_factory,
)
