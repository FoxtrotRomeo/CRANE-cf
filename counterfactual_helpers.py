from aeon.distances import dtw_distance, euclidean_distance, lcss_distance
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from tensorflow import keras
from collections import Counter

import os
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
import tensorflow as tf
from counterfactual_edits_helpers import (
    decode_ts_edits,
    apply_decoded_edits,
    EDIT_SLOT_DIM,
    TEXT_EDIT_SLOT_DIM,
)

os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

import torch
import keras
from keras import layers
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

def _validate_k(k: int) -> int:
    k_int = int(k)
    if k_int < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return k_int


def _normalize_selected_test_indices(selected_test_indices, n_test: int):
    idx = np.asarray(selected_test_indices, dtype=int).ravel()
    if idx.size == 0:
        return idx
    bad = idx[(idx < 0) | (idx >= int(n_test))]
    if bad.size > 0:
        raise IndexError(
            "selected_test_indices contains out-of-range values: "
            f"{np.unique(bad).tolist()} for n_test={int(n_test)}"
        )
    return idx


def _normalize_distance_metric(distance_metric, default="euclidean") -> str:
    metric = default if distance_metric is None else str(distance_metric).strip().lower()
    aliases = {
        "l2": "euclidean",
        "euclid": "euclidean",
        "cos": "cosine",
        "l1": "manhattan",
        "cityblock": "manhattan",
        "manhattan_distance": "manhattan",
        "hamming_distance": "hamming",
    }
    return aliases.get(metric, metric)


def _normalize_ts_distance_metric(distance_metric, default="dtw") -> str:
    metric = default if distance_metric is None else str(distance_metric).strip().lower()
    aliases = {
        "dynamic_time_warping": "dtw",
        "dtw_distance": "dtw",
        "euclid": "euclidean",
        "euclidean_distance": "euclidean",
        "ed": "euclidean",
        "lcss_distance": "lcss",
    }
    return aliases.get(metric, metric)


def _ts_channel_distance(
    x_f,
    y_f,
    *,
    distance_fn=None,
    distance_metric=None,
    dtw_window=0.10,
    distance_kwargs=None,
) -> float:
    """Univariate TS channel distance with metric-name and callable support."""
    kwargs = dict(distance_kwargs or {})

    if distance_metric is not None or distance_fn is None:
        metric = _normalize_ts_distance_metric(distance_metric, default="dtw")
        if metric == "dtw":
            if "window" not in kwargs:
                kwargs["window"] = dtw_window
            return float(dtw_distance(x_f, y_f, **kwargs))
        if metric == "euclidean":
            return float(euclidean_distance(x_f, y_f))
        if metric == "lcss":
            return float(lcss_distance(x_f, y_f, **kwargs))
        raise ValueError(
            f"Unsupported TS distance_metric '{distance_metric}'. "
            "Supported: 'dtw', 'euclidean', 'lcss'."
        )

    try:
        if "window" not in kwargs and dtw_window is not None:
            return float(distance_fn(x_f, y_f, window=dtw_window, **kwargs))
    except TypeError:
        pass

    try:
        return float(distance_fn(x_f, y_f, **kwargs))
    except TypeError:
        return float(distance_fn(x_f, y_f))


def _pairwise_vector_distances(
    query_samples,
    candidate_samples,
    *,
    distance_fn=None,
    distance_metric=None,
    default_metric="euclidean",
):
    """Robust pairwise distances for vector inputs: (n_query, n_cand)."""
    query_samples = np.asarray(query_samples)
    candidate_samples = np.asarray(candidate_samples)

    if query_samples.ndim == 1:
        query_samples = query_samples.reshape(1, -1)
    if candidate_samples.ndim == 1:
        candidate_samples = candidate_samples.reshape(1, -1)

    n_query = int(query_samples.shape[0])
    n_cand = int(candidate_samples.shape[0])

    if distance_metric is not None or distance_fn is None:
        metric = _normalize_distance_metric(distance_metric, default=default_metric)
        return np.asarray(
            pairwise_distances(query_samples, candidate_samples, metric=metric),
            dtype=float,
        )

    try:
        dm = np.asarray(distance_fn(query_samples, candidate_samples), dtype=float)
        if dm.shape == (n_query, n_cand):
            return dm
        if dm.shape == (n_cand, n_query):
            return dm.T
    except Exception:
        pass

    dist_matrix = np.empty((n_query, n_cand), dtype=float)
    for r in range(n_query):
        s = query_samples[r].reshape(1, -1)
        dists = None
        try:
            res = distance_fn(candidate_samples, s)
            dists = np.asarray(res, dtype=float).ravel()
            if dists.size != n_cand:
                try:
                    res2 = distance_fn(s, candidate_samples)
                    dists2 = np.asarray(res2, dtype=float).ravel()
                    dists = dists2 if dists2.size == n_cand else None
                except Exception:
                    dists = None
        except Exception:
            dists = None

        if dists is None or dists.size != n_cand:
            s1 = s.ravel()
            dists = np.empty(n_cand, dtype=float)
            for i in range(n_cand):
                try:
                    val = distance_fn(candidate_samples[i], s1)
                except Exception:
                    val = distance_fn(s1, candidate_samples[i])
                arr = np.asarray(val, dtype=float).ravel()
                dists[i] = float(arr[0] if arr.size > 0 else val)

        dist_matrix[r] = dists

    return dist_matrix


def find_k_closest_EF(
    X_train_static,
    X_train_meds,
    X_train_labs,
    X_train_text,
    y_train,
    X_test_static,
    X_test_meds,
    X_test_labs,
    X_test_text,
    selected_test_indices,
    target_value=0,
    k=5,
    e5_embed_fn=None,
    e5_tokenizer=None,
    e5_model=None,
    e5_device=None,
    distance_fn=euclidean_distances,
    distance_metric=None,
    return_indices=False,
):
    """
    For each test sample index in selected_test_indices, find the k closest samples in the concatenated multimodal space
    among those with y_train == target_value.

    The multimodal space is created by:
    - Averaging X_train_meds and X_train_labs over the time axis.
    - Generating E5 embeddings for X_train_text.
    - Concatenating: [X_train_static, averaged_meds, averaged_labs, e5_embeddings]

    Same for test data on selected_test_indices.

    Distances are computed using:
    - ``distance_metric`` (e.g., "euclidean", "manhattan", "hamming"), when provided, or
    - ``distance_fn`` callable otherwise (default: ``euclidean_distances``).

    For the output neighbors, a median approach is used:
    - neighbor 1: the closest neighbor
    - neighbor 2: median of the first 3 closest neighbors
    - neighbor 3: median of the first 5 closest neighbors
    - etc.

    Returns:
        neighbors_dict: dict mapping each selected test index -> numpy array of neighbor concatenated vectors
            (shape: (k_eff, concatenated_dim))
        neighbors_distances_dict: dict mapping each selected test index -> 1D numpy array of distances to the k closest (floats)
        If return_indices=True, also returns:
            neighbors_indices_dict: dict mapping each selected test index -> 1D numpy array of neighbor indices (length k_eff)
    """
    k = _validate_k(k)
    selected_test_indices = _normalize_selected_test_indices(selected_test_indices, X_test_static.shape[0])

    # Convert to arrays
    X_train_static = np.asarray(X_train_static)
    X_train_meds = np.asarray(X_train_meds)
    X_train_labs = np.asarray(X_train_labs)
    X_train_text = np.asarray(X_train_text, dtype=object).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1)
    X_test_static = np.asarray(X_test_static)
    X_test_meds = np.asarray(X_test_meds)
    X_test_labs = np.asarray(X_test_labs)
    X_test_text = np.asarray(X_test_text, dtype=object).reshape(-1)

    n_train = X_train_static.shape[0]
    n_test = X_test_static.shape[0]

    # Check shapes
    if X_train_meds.shape[0] != n_train or X_train_labs.shape[0] != n_train or X_train_text.shape[0] != n_train:
        raise ValueError("All train modality arrays must have the same number of samples.")
    if X_test_meds.shape[0] != n_test or X_test_labs.shape[0] != n_test or X_test_text.shape[0] != n_test:
        raise ValueError("All test modality arrays must have the same number of samples.")

    # Average time series
    X_train_meds_mean = np.mean(X_train_meds, axis=1)  # (n_train, d_meds)
    X_train_labs_mean = np.mean(X_train_labs, axis=1)  # (n_train, d_labs)
    X_test_meds_mean = np.mean(X_test_meds, axis=1)
    X_test_labs_mean = np.mean(X_test_labs, axis=1)

    # E5 embeddings
    if e5_embed_fn is None:
        if e5_tokenizer is None or e5_model is None or e5_device is None:
            raise ValueError("Provide e5_embed_fn or e5_tokenizer/e5_model/e5_device.")
        try:
            from counterfactual_evaluation_helpers import embed_e5
        except ImportError:
            from .counterfactual_evaluation_helpers import embed_e5

        def e5_embed_fn(texts):
            return embed_e5(texts, tokenizer=e5_tokenizer, model=e5_model, device=e5_device)

    train_texts_str = ["" if t is None else str(t) for t in X_train_text]
    test_texts_str = ["" if t is None else str(t) for t in X_test_text]
    emb_train = np.asarray(e5_embed_fn(train_texts_str), dtype=float)
    emb_test = np.asarray(e5_embed_fn(test_texts_str), dtype=float)

    # Concatenate train
    X_train_concat = np.concatenate([X_train_static, X_train_meds_mean, X_train_labs_mean, emb_train], axis=1)

    # For test, only selected
    selected_test_indices_arr = np.asarray(selected_test_indices, dtype=int)
    X_test_static_sel = X_test_static[selected_test_indices_arr]
    X_test_meds_mean_sel = X_test_meds_mean[selected_test_indices_arr]
    X_test_labs_mean_sel = X_test_labs_mean[selected_test_indices_arr]
    emb_test_sel = emb_test[selected_test_indices_arr]
    X_test_concat = np.concatenate([X_test_static_sel, X_test_meds_mean_sel, X_test_labs_mean_sel, emb_test_sel], axis=1)

    # Candidates
    candidate_indices = np.flatnonzero(y_train == target_value)
    candidate_samples = X_train_concat[candidate_indices]
    n_cand = candidate_samples.shape[0]

    if n_cand == 0:
        empty_neighbors = np.empty((0, X_train_concat.shape[1]), dtype=float)
        empty_dist = np.empty((0,), dtype=float)
        neighbors = {int(ti): empty_neighbors for ti in selected_test_indices}
        dists = {int(ti): empty_dist for ti in selected_test_indices}
        if return_indices:
            empty_idx = np.empty((0,), dtype=int)
            return neighbors, dists, {int(ti): empty_idx for ti in selected_test_indices}
        return neighbors, dists

    k_eff = min(k, n_cand)
    neighbors_dict = {}
    neighbors_distances_dict = {}
    neighbors_indices_dict = {} if return_indices else None

    # Compute distances for all selected test to all candidates
    dist_matrix = _pairwise_vector_distances(
        X_test_concat,
        candidate_samples,
        distance_fn=distance_fn,
        distance_metric=distance_metric,
        default_metric="euclidean",
    )

    for i, test_idx in enumerate(selected_test_indices_arr):
        dists = dist_matrix[i]

        # Select k_eff smallest distances
        if k_eff == n_cand:
            sel_local = np.argsort(dists)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, dists.shape[0] - 1))
            idx_k = np.argpartition(dists, kth)[:k_eff]
            sel_local = idx_k[np.argsort(dists[idx_k])]

        selected_train_indices = candidate_indices[sel_local].astype(int)

        # Create neighbors with median approach
        neighbors = np.empty((k_eff, X_train_concat.shape[1]), dtype=float)
        for j in range(k_eff):
            use_n = 2 * (j + 1) - 1  # 1, 3, 5, ...
            use_n = min(use_n, k_eff)
            subset = candidate_samples[sel_local[:use_n]]
            neighbors[j] = np.median(subset, axis=0)

        neighbors_dict[int(test_idx)] = neighbors
        neighbors_distances_dict[int(test_idx)] = dists[sel_local].astype(float)
        if return_indices:
            neighbors_indices_dict[int(test_idx)] = selected_train_indices

    if return_indices:
        return neighbors_dict, neighbors_distances_dict, neighbors_indices_dict
    return neighbors_dict, neighbors_distances_dict

def find_k_closest_EF_compressed(
    X_train_static,
    X_train_ts,
    y_train,
    X_test_static,
    X_test_ts,
    selected_test_indices,
    target_value=0,
    k=5,
    distance_fn=euclidean_distances,
    distance_metric=None,
    return_indices=False,
):
    """
    For each test sample index in selected_test_indices, find the k closest
    samples in a compressed early-fusion space:
    [X_static, mean_t(X_ts)], among those with y_train == target_value.
    Distances use ``distance_metric`` when provided (e.g., "euclidean", "manhattan", "hamming"),
    otherwise ``distance_fn``.
        Returns:
            neighbors_tabular_dict: dict mapping each selected test index -> numpy array of neighbor tabular vectors
                (shape: (k_eff, features_static + features_ts))
            neighbors_distances_dict: dict mapping each selected test index -> 1D numpy array of corresponding distances (floats)
            If return_indices=True, also returns:
                neighbors_indices_dict: dict mapping each selected test index -> 1D numpy array of neighbor indices (length k_eff)
    """
    k = _validate_k(k)
    selected_test_indices = _normalize_selected_test_indices(selected_test_indices, X_test_static.shape[0])

    # average the timeseries to get static representation
    X_train_ts_mean = np.mean(X_train_ts, axis=1)  # shape: (n_samples, features)
    X_test_ts_mean = np.mean(X_test_ts, axis=1)    # shape: (n_samples, features)

    # combine static and averaged timeseries representations
    X_train_combined = np.concatenate((X_train_static, X_train_ts_mean), axis=1)  # shape: (n_samples, features_static + features_ts)
    X_test_combined = np.concatenate((X_test_static, X_test_ts_mean), axis=1)  # shape: (n_samples, features_static + features_ts)

    # candidate indices in X_train_combined that have the requested label
    candidate_indices = np.flatnonzero(y_train == target_value)
    candidate_samples = X_train_combined[candidate_indices]  # shape (n_candidates, features)
    n_cand = candidate_samples.shape[0]
    if n_cand == 0:
        empty_neighbors = X_train_combined[:0]
        empty_dist = np.empty((0,), dtype=float)
        neighbors = {int(ti): empty_neighbors for ti in selected_test_indices}
        dists = {int(ti): empty_dist for ti in selected_test_indices}
        if return_indices:
            empty_idx = np.empty((0,), dtype=int)
            return neighbors, dists, {int(ti): empty_idx for ti in selected_test_indices}
        return neighbors, dists
    k_eff = min(k, n_cand)
    neighbors_tabular_dict = {}
    neighbors_distances_dict = {}
    neighbors_indices_dict = {} if return_indices else None
    selected_test_indices_arr = np.asarray(selected_test_indices, dtype=int)
    dist_matrix = _pairwise_vector_distances(
        X_test_combined[selected_test_indices_arr],
        candidate_samples,
        distance_fn=distance_fn,
        distance_metric=distance_metric,
        default_metric="euclidean",
    )
    for r, test_idx in enumerate(selected_test_indices_arr):
        dists = dist_matrix[r]

        # get k_eff smallest distances (unsorted via argpartition then sorted)
        if k_eff == n_cand:
            sel_local = np.argsort(dists)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, dists.shape[0] - 1))
            idx_k = np.argpartition(dists, kth)[:k_eff]
            sel_local = idx_k[np.argsort(dists[idx_k])]
        selected_train_indices = candidate_indices[sel_local].astype(int)
        neighbors_tabular_dict[int(test_idx)] = X_train_combined[selected_train_indices]
        neighbors_distances_dict[int(test_idx)] = dists[sel_local].astype(float)
        if return_indices:
            neighbors_indices_dict[int(test_idx)] = selected_train_indices
    if return_indices:
        return neighbors_tabular_dict, neighbors_distances_dict, neighbors_indices_dict
    return neighbors_tabular_dict, neighbors_distances_dict


def find_k_closest_static(
    X_train_static,
    y_train,
    X_test_static,
    selected_test_indices,
    target_value=1,
    k=50,
    distance_fn=euclidean_distances,
    distance_metric=None,
    return_indices=False,
):
    """
    For each test sample index in selected_test_indices, find the k closest samples in X_train_static
    among those with y_train == target_value using either:
      - ``distance_metric`` (e.g., "euclidean", "manhattan", "hamming"), when provided, or
      - ``distance_fn`` otherwise.

    Returns:
        neighbors_indices_dict: dict mapping each selected test index -> 1D numpy array of neighbor indices (k_eff,)
        If return_indices=True, also returns:
            neighbors_distances_dict: dict mapping each selected test index -> 1D numpy array of distances (k_eff,)
            (distance from the test sample to each selected neighbor)
    """
    k = _validate_k(k)
    selected_test_indices = _normalize_selected_test_indices(selected_test_indices, X_test_static.shape[0])

    # candidate indices in X_train_static that have the requested label
    candidate_indices = np.flatnonzero(y_train == target_value)
    candidate_samples = X_train_static[candidate_indices]  # shape (n_candidates, features)

    n_cand = candidate_samples.shape[0]
    if n_cand == 0:
        empty_idx = np.empty((0,), dtype=int)
        neighbors_indices_dict = {int(ti): empty_idx for ti in selected_test_indices}
        if return_indices:
            empty_dist = np.empty((0,), dtype=float)
            neighbors_distances_dict = {int(ti): empty_dist for ti in selected_test_indices}
            return neighbors_indices_dict, neighbors_distances_dict
        return neighbors_indices_dict

    k_eff = min(k, n_cand)
    neighbors_indices_dict = {}
    neighbors_distances_dict = {} if return_indices else None

    selected_test_indices_arr = np.asarray(selected_test_indices, dtype=int)
    dist_matrix = _pairwise_vector_distances(
        X_test_static[selected_test_indices_arr],
        candidate_samples,
        distance_fn=distance_fn,
        distance_metric=distance_metric,
        default_metric="euclidean",
    )

    for r, test_idx in enumerate(selected_test_indices_arr):
        dists = dist_matrix[r]

        # get k_eff smallest distances (unsorted via argpartition then sorted)
        if k_eff == n_cand:
            sel_local = np.argsort(dists)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, dists.shape[0] - 1))
            idx_k = np.argpartition(dists, kth)[:k_eff]
            sel_local = idx_k[np.argsort(dists[idx_k])]

        selected_train_indices = candidate_indices[sel_local].astype(int)
        neighbors_indices_dict[int(test_idx)] = selected_train_indices

        if return_indices:
            neighbors_distances_dict[int(test_idx)] = dists[sel_local].astype(float)

    if return_indices:
        return neighbors_indices_dict, neighbors_distances_dict
    return neighbors_indices_dict

def find_k_closest_ts(
    X_train_ts,
    y_train,
    X_test_ts,
    selected_test_indices,
    target_value=0,
    k=5,
    distance_fn=dtw_distance,
    distance_metric=None,
    dtw_window=0.10,
    distance_kwargs=None,
    return_indices=False,
):
    """
    For each test sample index in selected_test_indices, find the k closest samples in X_train_ts
    among those with y_train == target_value using either:
      - ``distance_metric`` in {"dtw", "euclidean", "lcss"}, or
      - ``distance_fn`` callable (backward compatible).

    ``dtw_window`` controls the DTW warping window when DTW is used (string metric
    mode, or callable mode when the callable accepts ``window``).
    ``distance_kwargs`` forwards optional extra keyword arguments to the selected
    distance implementation (e.g., ``{"epsilon": 0.5}`` for LCSS).

    Returns:
        neighbors_indices_dict: dict mapping each selected test index -> 1D numpy array of neighbor indices (k_eff,)
        If return_indices=True, also returns:
            neighbors_distances_dict: dict mapping each selected test index -> 1D numpy array of distances (k_eff,)
            (distance from the test sample to each selected neighbor)
    """
    k = _validate_k(k)
    selected_test_indices = _normalize_selected_test_indices(selected_test_indices, X_test_ts.shape[0])

    # candidate indices in X_train_ts that have the requested label
    candidate_indices = np.flatnonzero(y_train == target_value)
    candidate_samples = X_train_ts[candidate_indices]  # shape (n_candidates, timesteps, features)

    n_cand = candidate_samples.shape[0]
    if n_cand == 0:
        empty_idx = np.empty((0,), dtype=int)
        neighbors_indices_dict = {int(ti): empty_idx for ti in selected_test_indices}
        if return_indices:
            empty_dist = np.empty((0,), dtype=float)
            neighbors_distances_dict = {int(ti): empty_dist for ti in selected_test_indices}
            return neighbors_indices_dict, neighbors_distances_dict
        return neighbors_indices_dict

    k_eff = min(k, n_cand)
    neighbors_indices_dict = {}
    neighbors_distances_dict = {} if return_indices else None

    for test_idx in selected_test_indices:
        s = X_test_ts[test_idx]  # shape: (timesteps, features)
        _, n_features = s.shape

        # distances to all candidates (sum distance per feature)
        dists = np.empty(n_cand, dtype=float)
        for j in range(n_cand):
            cand = candidate_samples[j]  # shape: (timesteps, features)
            total = 0.0
            for f in range(n_features):
                x_f = np.ascontiguousarray(s[:, f])
                y_f = np.ascontiguousarray(cand[:, f])
                total += _ts_channel_distance(
                    x_f,
                    y_f,
                    distance_fn=distance_fn,
                    distance_metric=distance_metric,
                    dtw_window=dtw_window,
                    distance_kwargs=distance_kwargs,
                )
            dists[j] = total

        # get k_eff smallest distances (unsorted via argpartition then sorted)
        if k_eff == n_cand:
            sel_local = np.argsort(dists)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, dists.shape[0] - 1))
            idx_k = np.argpartition(dists, kth)[:k_eff]
            sel_local = idx_k[np.argsort(dists[idx_k])]

        selected_train_indices = candidate_indices[sel_local].astype(int)
        neighbors_indices_dict[int(test_idx)] = selected_train_indices

        if return_indices:
            neighbors_distances_dict[int(test_idx)] = dists[sel_local].astype(float)

    if return_indices:
        return neighbors_indices_dict, neighbors_distances_dict
    return neighbors_indices_dict


def _normalize_text_encoder(text_encoder, default="e5") -> str:
    enc = default if text_encoder is None else str(text_encoder).strip().lower()
    aliases = {
        "tf-idf": "tfidf",
        "word2vec_avg": "word2vec",
        "w2v": "word2vec",
        "sbert": "bert",
        "sentence-bert": "bert",
        "raw_text": "raw",
    }
    return aliases.get(enc, enc)


def _normalize_text_distance_metric(text_distance_metric, default="cosine") -> str:
    metric = default if text_distance_metric is None else str(text_distance_metric).strip().lower()
    aliases = {
        "cos": "cosine",
        "l2": "euclidean",
        "l1": "manhattan",
        "rougel": "rouge_l",
        "rouge-l": "rouge_l",
        "rouge_1": "rouge1",
        "rouge-1": "rouge1",
        "rouge_2": "rouge2",
        "rouge-2": "rouge2",
    }
    return aliases.get(metric, metric)


def _to_text_list(text_array) -> list:
    arr = np.asarray(text_array, dtype=object).reshape(-1)
    return ["" if t is None else str(t) for t in arr]


def _tokenize_text(text: str) -> list:
    s = "" if text is None else str(text)
    return [tok for tok in s.lower().split() if tok]


def _lcs_length(tokens_a: list, tokens_b: list) -> int:
    if not tokens_a or not tokens_b:
        return 0
    dp = [0] * (len(tokens_b) + 1)
    for ta in tokens_a:
        prev = 0
        for j, tb in enumerate(tokens_b, start=1):
            old = dp[j]
            if ta == tb:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = old
    return int(dp[-1])


def _ngram_counter(tokens: list, n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _bleu_similarity(reference_tokens: list, candidate_tokens: list, max_n: int = 4, smooth: float = 1e-12) -> float:
    if len(candidate_tokens) == 0:
        return 1.0 if len(reference_tokens) == 0 else 0.0

    max_order = max(1, min(int(max_n), len(candidate_tokens), max(len(reference_tokens), 1)))
    precisions = []
    for n in range(1, max_order + 1):
        cand_counts = _ngram_counter(candidate_tokens, n)
        ref_counts = _ngram_counter(reference_tokens, n)
        total = sum(cand_counts.values())
        overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
        p_n = (overlap + smooth) / (total + smooth)
        precisions.append(max(float(p_n), smooth))

    log_precision = float(np.mean(np.log(precisions)))

    ref_len = len(reference_tokens)
    cand_len = len(candidate_tokens)
    if cand_len > ref_len:
        brevity_penalty = 1.0
    else:
        brevity_penalty = float(np.exp(1.0 - (ref_len / max(cand_len, 1))))

    score = brevity_penalty * float(np.exp(log_precision))
    return float(np.clip(score, 0.0, 1.0))


def _rouge_n_f1_similarity(reference_tokens: list, candidate_tokens: list, n: int) -> float:
    ref_counts = _ngram_counter(reference_tokens, n)
    cand_counts = _ngram_counter(candidate_tokens, n)
    if len(ref_counts) == 0 and len(cand_counts) == 0:
        return 1.0
    if len(ref_counts) == 0 or len(cand_counts) == 0:
        return 0.0

    overlap = sum(min(count, ref_counts.get(gram, 0)) for gram, count in cand_counts.items())
    cand_total = max(sum(cand_counts.values()), 1)
    ref_total = max(sum(ref_counts.values()), 1)
    precision = overlap / cand_total
    recall = overlap / ref_total
    if precision + recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _rouge_l_f1_similarity(reference_tokens: list, candidate_tokens: list) -> float:
    if len(reference_tokens) == 0 and len(candidate_tokens) == 0:
        return 1.0
    if len(reference_tokens) == 0 or len(candidate_tokens) == 0:
        return 0.0
    lcs = _lcs_length(reference_tokens, candidate_tokens)
    precision = lcs / max(len(candidate_tokens), 1)
    recall = lcs / max(len(reference_tokens), 1)
    if precision + recall <= 0.0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def _lcs_similarity(tokens_a: list, tokens_b: list) -> float:
    if len(tokens_a) == 0 and len(tokens_b) == 0:
        return 1.0
    denom = max(len(tokens_a), len(tokens_b), 1)
    return float(_lcs_length(tokens_a, tokens_b) / denom)


def _direct_text_distance(query_text: str, candidate_text: str, distance_metric: str, metric_kwargs: dict) -> float:
    metric = _normalize_text_distance_metric(distance_metric, default="rouge_l")
    q_tokens = _tokenize_text(query_text)
    c_tokens = _tokenize_text(candidate_text)

    if metric == "bleu":
        sim = _bleu_similarity(
            reference_tokens=q_tokens,
            candidate_tokens=c_tokens,
            max_n=int(metric_kwargs.get("max_n", 4)),
            smooth=float(metric_kwargs.get("smooth", 1e-12)),
        )
    elif metric == "rouge1":
        sim = _rouge_n_f1_similarity(q_tokens, c_tokens, n=1)
    elif metric == "rouge2":
        sim = _rouge_n_f1_similarity(q_tokens, c_tokens, n=2)
    elif metric == "rouge_l":
        sim = _rouge_l_f1_similarity(q_tokens, c_tokens)
    elif metric == "lcs":
        sim = _lcs_similarity(q_tokens, c_tokens)
    else:
        raise ValueError(
            f"Unsupported direct text distance metric '{distance_metric}'. "
            "Use one of {'bleu', 'rouge1', 'rouge2', 'rouge_l', 'lcs'}."
        )

    return float(1.0 - np.clip(sim, 0.0, 1.0))


def _pairwise_text_distances(
    query_texts,
    candidate_texts,
    *,
    distance_metric="rouge_l",
    distance_fn=None,
    metric_kwargs=None,
):
    query_arr = np.asarray(query_texts, dtype=object).reshape(-1)
    cand_arr = np.asarray(candidate_texts, dtype=object).reshape(-1)
    n_query = int(query_arr.shape[0])
    n_cand = int(cand_arr.shape[0])
    metric_kwargs = dict(metric_kwargs or {})

    if distance_fn is not None:
        try:
            dm = np.asarray(distance_fn(query_arr, cand_arr), dtype=float)
            if dm.shape == (n_query, n_cand):
                return dm
            if dm.shape == (n_cand, n_query):
                return dm.T
        except Exception:
            pass

    out = np.empty((n_query, n_cand), dtype=float)
    for i, q in enumerate(query_arr):
        for j, c in enumerate(cand_arr):
            if distance_fn is not None:
                try:
                    out[i, j] = float(distance_fn(q, c))
                except Exception:
                    out[i, j] = float(distance_fn(str(q), str(c)))
            else:
                out[i, j] = _direct_text_distance(str(q), str(c), distance_metric, metric_kwargs)
    return out


def _embed_bert_mean_pool(
    texts,
    *,
    tokenizer,
    model,
    device=None,
    batch_size=32,
    max_length=512,
):
    if tokenizer is None or model is None:
        raise ValueError("BERT encoding requires tokenizer and model.")

    device_obj = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device_obj)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch = texts[start : start + int(batch_size)]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(device_obj) for k, v in enc.items()}
            out = model(**enc)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-12)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=float)


def _embed_word2vec_average(texts, *, word2vec_model):
    if word2vec_model is None:
        raise ValueError("word2vec encoding requires word2vec_model or word2vec_embed_fn.")

    wv = getattr(word2vec_model, "wv", word2vec_model)
    vector_size = getattr(wv, "vector_size", None)
    if vector_size is None:
        raise ValueError("word2vec_model must expose vector_size (directly or via .wv).")

    out = np.zeros((len(texts), int(vector_size)), dtype=float)
    for i, t in enumerate(texts):
        tokens = _tokenize_text(t)
        vecs = []
        for tok in tokens:
            try:
                vecs.append(np.asarray(wv[tok], dtype=float))
            except Exception:
                continue
        if vecs:
            out[i] = np.mean(np.vstack(vecs), axis=0)
    return out


def _build_text_representations(
    train_texts_str,
    test_texts_str,
    *,
    text_encoder="e5",
    text_embed_fn=None,
    e5_embed_fn=None,
    e5_tokenizer=None,
    e5_model=None,
    e5_device=None,
    bert_embed_fn=None,
    bert_tokenizer=None,
    bert_model=None,
    bert_device=None,
    bert_batch_size=32,
    bert_max_length=512,
    tfidf_vectorizer=None,
    tfidf_kwargs=None,
    word2vec_embed_fn=None,
    word2vec_model=None,
):
    enc = _normalize_text_encoder(text_encoder, default="e5")

    if enc == "raw":
        raise ValueError("Raw-text encoder does not produce vectors. Use direct text metrics.")

    if enc == "tfidf":
        vec = tfidf_vectorizer if tfidf_vectorizer is not None else TfidfVectorizer(**(tfidf_kwargs or {}))
        emb_train = vec.fit_transform(train_texts_str)
        emb_test = vec.transform(test_texts_str)
        emb_train = emb_train.toarray() if hasattr(emb_train, "toarray") else np.asarray(emb_train)
        emb_test = emb_test.toarray() if hasattr(emb_test, "toarray") else np.asarray(emb_test)
        return np.asarray(emb_train, dtype=float), np.asarray(emb_test, dtype=float)

    if enc == "word2vec":
        embed_fn = text_embed_fn or word2vec_embed_fn
        if embed_fn is not None:
            emb_train = np.asarray(embed_fn(train_texts_str), dtype=float)
            emb_test = np.asarray(embed_fn(test_texts_str), dtype=float)
        else:
            emb_train = _embed_word2vec_average(train_texts_str, word2vec_model=word2vec_model)
            emb_test = _embed_word2vec_average(test_texts_str, word2vec_model=word2vec_model)
        return emb_train, emb_test

    if enc == "bert":
        embed_fn = text_embed_fn or bert_embed_fn
        if embed_fn is not None:
            emb_train = np.asarray(embed_fn(train_texts_str), dtype=float)
            emb_test = np.asarray(embed_fn(test_texts_str), dtype=float)
        else:
            emb_train = _embed_bert_mean_pool(
                train_texts_str,
                tokenizer=bert_tokenizer,
                model=bert_model,
                device=bert_device,
                batch_size=bert_batch_size,
                max_length=bert_max_length,
            )
            emb_test = _embed_bert_mean_pool(
                test_texts_str,
                tokenizer=bert_tokenizer,
                model=bert_model,
                device=bert_device,
                batch_size=bert_batch_size,
                max_length=bert_max_length,
            )
        return emb_train, emb_test

    if enc in {"e5", "custom"}:
        embed_fn = text_embed_fn or e5_embed_fn
        if embed_fn is None and enc == "e5":
            if e5_tokenizer is None or e5_model is None or e5_device is None:
                raise ValueError(
                    "E5 encoding requires e5_embed_fn or e5_tokenizer/e5_model/e5_device."
                )
            try:
                from counterfactual_evaluation_helpers import embed_e5
            except ImportError:  # support package-style imports
                from .counterfactual_evaluation_helpers import embed_e5

            def embed_fn(texts):
                return embed_e5(
                    texts,
                    tokenizer=e5_tokenizer,
                    model=e5_model,
                    device=e5_device,
                )
        if embed_fn is None:
            raise ValueError("Custom text encoder requires text_embed_fn.")
        emb_train = np.asarray(embed_fn(train_texts_str), dtype=float)
        emb_test = np.asarray(embed_fn(test_texts_str), dtype=float)
        return emb_train, emb_test

    raise ValueError(
        f"Unsupported text_encoder '{text_encoder}'. "
        "Use one of {'e5', 'bert', 'tfidf', 'word2vec', 'custom', 'raw'}."
    )


def find_k_closest_text(
    X_train_static,
    X_train_meds,
    X_train_labs,
    train_text_for_distance,
    y_train,
    X_test_static,
    X_test_meds,
    X_test_labs,
    test_text_for_distance,
    selected_test_indices,
    *,
    target_value=0,
    k=5,
    text_encoder="e5",
    text_distance_metric="cosine",
    text_distance_fn=None,
    text_metric_kwargs=None,
    text_embed_fn=None,
    e5_embed_fn=None,
    e5_tokenizer=None,
    e5_model=None,
    e5_device=None,
    bert_embed_fn=None,
    bert_tokenizer=None,
    bert_model=None,
    bert_device=None,
    bert_batch_size=32,
    bert_max_length=512,
    tfidf_vectorizer=None,
    tfidf_kwargs=None,
    word2vec_embed_fn=None,
    word2vec_model=None,
    train_text_raw=None,
    test_text_raw=None,
):
    """Text-NN counterfactuals with configurable encoders and distance metrics.

    Supported encoders:
      - ``"e5"`` (default), ``"bert"``, ``"tfidf"``, ``"word2vec"``,
      - ``"custom"`` (via ``text_embed_fn``),
      - ``"raw"`` (for direct text metrics).

    Supported direct text metrics (operating on raw strings):
      - ``"bleu"``, ``"rouge1"``, ``"rouge2"``, ``"rouge_l"``, ``"lcs"``.

    For vector encoders, distances are computed with ``text_distance_metric`` via
    pairwise distances (e.g., cosine, euclidean, manhattan, hamming), or with
    ``text_distance_fn`` when provided.
    """
    k = _validate_k(k)
    selected_test_indices = _normalize_selected_test_indices(
        selected_test_indices, np.asarray(X_test_static).shape[0]
    )

    X_train_static = np.asarray(X_train_static)
    X_test_static = np.asarray(X_test_static)
    y_train = np.asarray(y_train).reshape(-1)

    n_train = int(X_train_static.shape[0])
    n_test = int(X_test_static.shape[0])

    if X_train_meds is None:
        X_train_meds = np.zeros((n_train, 1, 1), dtype=float)
    if X_test_meds is None:
        X_test_meds = np.zeros((n_test, 1, 1), dtype=float)
    if X_train_labs is None:
        X_train_labs = np.zeros((n_train, 1, 1), dtype=float)
    if X_test_labs is None:
        X_test_labs = np.zeros((n_test, 1, 1), dtype=float)

    X_train_meds = np.asarray(X_train_meds)
    X_test_meds = np.asarray(X_test_meds)
    X_train_labs = np.asarray(X_train_labs)
    X_test_labs = np.asarray(X_test_labs)

    if (
        n_train != int(X_train_meds.shape[0])
        or n_train != int(X_train_labs.shape[0])
        or n_train != int(y_train.shape[0])
    ):
        raise ValueError(
            "X_train_static, X_train_meds, X_train_labs, and y_train must have matching first dimension."
        )
    if n_test != int(X_test_meds.shape[0]) or n_test != int(X_test_labs.shape[0]):
        raise ValueError(
            "X_test_static, X_test_meds, and X_test_labs must have matching first dimension."
        )

    train_text_arr = np.asarray(train_text_for_distance, dtype=object).reshape(-1)
    test_text_arr = np.asarray(test_text_for_distance, dtype=object).reshape(-1)
    if train_text_arr.shape[0] != X_train_static.shape[0]:
        raise ValueError("train_text_for_distance length must match training rows.")
    if test_text_arr.shape[0] != X_test_static.shape[0]:
        raise ValueError("test_text_for_distance length must match test rows.")

    if train_text_raw is None:
        train_text_raw = train_text_arr
    else:
        train_text_raw = np.asarray(train_text_raw, dtype=object).reshape(-1)
    if test_text_raw is None:
        test_text_raw = test_text_arr
    else:
        test_text_raw = np.asarray(test_text_raw, dtype=object).reshape(-1)

    if train_text_raw.shape[0] != train_text_arr.shape[0]:
        raise ValueError("train_text_raw length must match train_text_for_distance length.")
    if test_text_raw.shape[0] != test_text_arr.shape[0]:
        raise ValueError("test_text_raw length must match test_text_for_distance length.")

    train_texts_str = _to_text_list(train_text_arr)
    test_texts_str = _to_text_list(test_text_arr)

    candidate_indices = np.flatnonzero(y_train == target_value).astype(int)
    n_cand = candidate_indices.shape[0]

    text_nn_cfs_dict = {}
    neighbors_indices_dict = {}
    neighbors_distances_dict = {}

    if n_cand == 0:
        empty_idx = np.empty((0,), dtype=int)
        empty_dist = np.empty((0,), dtype=float)
        for ti in selected_test_indices:
            text_nn_cfs_dict[int(ti)] = []
            neighbors_indices_dict[int(ti)] = empty_idx
            neighbors_distances_dict[int(ti)] = empty_dist
        return text_nn_cfs_dict, neighbors_indices_dict, neighbors_distances_dict

    selected_test_indices_arr = np.asarray(selected_test_indices, dtype=int)
    metric_norm = _normalize_text_distance_metric(text_distance_metric, default="cosine")
    text_metric_kwargs = dict(text_metric_kwargs or {})

    direct_text_metrics = {"bleu", "rouge1", "rouge2", "rouge_l", "lcs"}
    encoder_norm = _normalize_text_encoder(text_encoder, default="e5")

    if metric_norm in direct_text_metrics or encoder_norm == "raw":
        query_texts = np.asarray(test_texts_str, dtype=object)[selected_test_indices_arr]
        candidate_texts = np.asarray(train_texts_str, dtype=object)[candidate_indices]
        dist_matrix = _pairwise_text_distances(
            query_texts,
            candidate_texts,
            distance_metric=metric_norm if metric_norm in direct_text_metrics else "rouge_l",
            distance_fn=text_distance_fn,
            metric_kwargs=text_metric_kwargs,
        )
    else:
        emb_train, emb_test = _build_text_representations(
            train_texts_str,
            test_texts_str,
            text_encoder=encoder_norm,
            text_embed_fn=text_embed_fn,
            e5_embed_fn=e5_embed_fn,
            e5_tokenizer=e5_tokenizer,
            e5_model=e5_model,
            e5_device=e5_device,
            bert_embed_fn=bert_embed_fn,
            bert_tokenizer=bert_tokenizer,
            bert_model=bert_model,
            bert_device=bert_device,
            bert_batch_size=bert_batch_size,
            bert_max_length=bert_max_length,
            tfidf_vectorizer=tfidf_vectorizer,
            tfidf_kwargs=tfidf_kwargs,
            word2vec_embed_fn=word2vec_embed_fn,
            word2vec_model=word2vec_model,
        )
        if emb_train.ndim != 2 or emb_test.ndim != 2:
            raise ValueError("Text representations must be 2D arrays: (N, D).")
        if emb_train.shape[1] != emb_test.shape[1]:
            raise ValueError("Train/test representation dimensions must match.")
        if emb_train.shape[0] != n_train or emb_test.shape[0] != n_test:
            raise ValueError("Text representation rows must match train/test sample counts.")

        dist_matrix = _pairwise_vector_distances(
            emb_test[selected_test_indices_arr],
            emb_train[candidate_indices],
            distance_fn=text_distance_fn,
            distance_metric=metric_norm,
            default_metric="cosine",
        )

    k_eff = min(k, n_cand)
    for r, test_idx in enumerate(selected_test_indices_arr):
        dists = dist_matrix[r]

        if k_eff == n_cand:
            sel_local = np.argsort(dists)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, dists.shape[0] - 1))
            idx_k = np.argpartition(dists, kth)[:k_eff]
            sel_local = idx_k[np.argsort(dists[idx_k])]

        ordered_train_idx = candidate_indices[sel_local].astype(int)
        ordered_dists = dists[sel_local].astype(float)

        cfs = []
        for i in range(1, k_eff + 1):
            use_n = min(2 * i - 1, k_eff)
            subset_idx = ordered_train_idx[:use_n]
            anchor_idx = int(ordered_train_idx[use_n - 1])

            tab_med = np.median(X_train_static[subset_idx], axis=0)
            meds_med = np.median(X_train_meds[subset_idx], axis=0)
            labs_med = np.median(X_train_labs[subset_idx], axis=0)

            cfs.append(
                {
                    "static": np.asarray(tab_med, dtype=float),
                    "meds": np.asarray(meds_med, dtype=float),
                    "labs": np.asarray(labs_med, dtype=float),
                    "tab": np.asarray(tab_med, dtype=float),
                    "ts": {
                        "labs": np.asarray(labs_med, dtype=float),
                        "meds": np.asarray(meds_med, dtype=float),
                    },
                    "text": str(train_text_arr[anchor_idx]),
                    "text_input": train_text_raw[anchor_idx],
                    "source_train_idx": anchor_idx,
                    "n_neighbors_used": int(use_n),
                }
            )

        text_nn_cfs_dict[int(test_idx)] = cfs
        neighbors_indices_dict[int(test_idx)] = ordered_train_idx
        neighbors_distances_dict[int(test_idx)] = ordered_dists

    return text_nn_cfs_dict, neighbors_indices_dict, neighbors_distances_dict


def from_indices_to_neighbors(
    indices_dict,
    train_static,
    train_meds=None,
    train_labs=None,
    train_text=None,
):
    """
    Given a dictionary mapping test indices to arrays of neighbor indices (assumed sorted by closeness),
    return reconstructed neighbors.

    Multimodal mode (preferred, when meds/labs/text are provided):
    Construction rule (for i = 1..n_neighbors):
        - neighbor 1: the single nearest neighbor (median of first 1)
        - neighbor 2: median of the first 3 neighbors
        - neighbor 3: median of the first 5 neighbors
        - ...
        - in general: neighbor i is the median of the first (2*i - 1) neighbors
      For text, no interpolation is used: the text is copied from the anchor
      neighbor at rank (2*i - 1), clipped to available neighbors.

    Legacy mode (when only train_static is provided):
      Preserves the old behavior and returns numpy arrays reconstructed from
      train_static only.

    Args:
        indices_dict: dict mapping test index -> 1D numpy array of neighbor indices (length n_neighbors)
        train_static: numpy array of shape (n_samples, D_static)
        train_meds: numpy array of shape (n_samples, T, D_meds)
        train_labs: numpy array of shape (n_samples, T, D_labs)
        train_text: array-like length n_samples (raw text inputs/strings)

    Returns:
        neighbors_dict:
          - multimodal mode: dict[test_idx -> list[dict]] where each dict has
            keys 'static','meds','labs','text' plus compatibility keys
            'tab','ts','text_input'
          - legacy mode: dict[test_idx -> np.ndarray], same as old implementation
    """
    # Legacy fallback for older calls.
    if train_meds is None or train_labs is None or train_text is None:
        train_data = np.asarray(train_static)
        neighbors_dict = {}
        tail_shape = train_data.shape[1:]

        for test_idx, neighbor_indices in indices_dict.items():
            idx = np.asarray(neighbor_indices, dtype=int)
            n_avail = int(idx.size)

            if n_avail == 0:
                neighbors_dict[int(test_idx)] = np.empty((0,) + tail_shape, dtype=float)
                continue

            out = np.empty((n_avail,) + tail_shape, dtype=float)
            for i in range(1, n_avail + 1):
                use_n = min(2 * i - 1, n_avail)
                block = train_data[idx[:use_n]]
                out[i - 1] = np.median(block, axis=0)
            neighbors_dict[int(test_idx)] = out
        return neighbors_dict

    train_static = np.asarray(train_static)
    train_meds = np.asarray(train_meds)
    train_labs = np.asarray(train_labs)
    train_text = np.asarray(train_text, dtype=object).reshape(-1)

    n_train = int(train_static.shape[0])
    if train_meds.shape[0] != n_train or train_labs.shape[0] != n_train or train_text.shape[0] != n_train:
        raise ValueError("train_static, train_meds, train_labs, and train_text must align on first dimension.")

    neighbors_dict = {}

    for test_idx, neighbor_indices in indices_dict.items():
        idx = np.asarray(neighbor_indices, dtype=int)
        n_avail = int(idx.size)

        if n_avail == 0:
            neighbors_dict[int(test_idx)] = []
            continue

        out = []
        for i in range(1, n_avail + 1):
            use_n = min(2 * i - 1, n_avail)
            subset_idx = idx[:use_n]
            anchor_idx = int(idx[use_n - 1])

            static_med = np.median(train_static[subset_idx], axis=0)
            meds_med = np.median(train_meds[subset_idx], axis=0)
            labs_med = np.median(train_labs[subset_idx], axis=0)

            out.append(
                {
                    "static": np.asarray(static_med, dtype=float),
                    "meds": np.asarray(meds_med, dtype=float),
                    "labs": np.asarray(labs_med, dtype=float),
                    "text": str(train_text[anchor_idx]),
                    "tab": np.asarray(static_med, dtype=float),
                    "ts": {
                        "labs": np.asarray(labs_med, dtype=float),
                        "meds": np.asarray(meds_med, dtype=float),
                    },
                    "text_input": train_text[anchor_idx],
                    "source_train_idx": anchor_idx,
                    "n_neighbors_used": int(use_n),
                }
            )

        neighbors_dict[int(test_idx)] = out

    return neighbors_dict

def find_k_closest_combined(
    indices_static_dict,
    distances_static_dict,
    indices_meds_dict,
    distances_meds_dict,
    indices_labs_dict,
    distances_labs_dict,
    indices_text_dict,
    distances_text_dict,
    k=5,
    return_distances=False,
):
    """
    Given four dictionaries of neighbor indices and distances
    (from static, meds, labs, and text parts),
    for each test sample:
      1) intersect the neighbor indices from all four modalities
      2) sum distances for each intersected index
      3) select the k indices with the lowest summed distance

    Returns:
        combined_indices_dict: dict mapping test index -> 1D numpy array of neighbor indices (k_eff,)
        If return_distances=True, also returns:
            combined_distances_dict: dict mapping test index -> 1D numpy array of summed distances (k_eff,)
    """
    k = _validate_k(k)
    common_test_indices = (
        set(indices_static_dict.keys())
        .intersection(indices_meds_dict.keys())
        .intersection(indices_labs_dict.keys())
        .intersection(indices_text_dict.keys())
    )

    combined_indices_dict = {}
    combined_distances_dict = {} if return_distances else None

    for test_idx in common_test_indices:
        idx_static = np.asarray(indices_static_dict[test_idx], dtype=int).ravel()
        d_static = np.asarray(distances_static_dict[test_idx], dtype=float).ravel()

        idx_meds = np.asarray(indices_meds_dict[test_idx], dtype=int).ravel()
        d_meds = np.asarray(distances_meds_dict[test_idx], dtype=float).ravel()

        idx_labs = np.asarray(indices_labs_dict[test_idx], dtype=int).ravel()
        d_labs = np.asarray(distances_labs_dict[test_idx], dtype=float).ravel()

        idx_text = np.asarray(indices_text_dict[test_idx], dtype=int).ravel()
        d_text = np.asarray(distances_text_dict[test_idx], dtype=float).ravel()

        # Map neighbor index -> distance
        map_static = {int(i): float(d) for i, d in zip(idx_static, d_static)}
        map_meds = {int(i): float(d) for i, d in zip(idx_meds, d_meds)}
        map_labs = {int(i): float(d) for i, d in zip(idx_labs, d_labs)}
        map_text = {int(i): float(d) for i, d in zip(idx_text, d_text)}

        # Intersection of indices for this test sample
        common_candidates = np.array(
            sorted(
                set(map_static.keys())
                .intersection(map_meds.keys())
                .intersection(map_labs.keys())
                .intersection(map_text.keys())
            ),
            dtype=int,
        )

        if common_candidates.size == 0:
            combined_indices_dict[int(test_idx)] = np.empty((0,), dtype=int)
            if return_distances:
                combined_distances_dict[int(test_idx)] = np.empty((0,), dtype=float)
            continue

        summed_distances = np.array(
            [
                map_static[int(i)]
                + map_meds[int(i)]
                + map_labs[int(i)]
                + map_text[int(i)]
                for i in common_candidates
            ],
            dtype=float,
        )

        n = common_candidates.size
        k_eff = min(k, int(n))

        if k_eff == n:
            sel = np.argsort(summed_distances)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, summed_distances.shape[0] - 1))
            idx_k = np.argpartition(summed_distances, kth)[:k_eff]
            sel = idx_k[np.argsort(summed_distances[idx_k])]

        combined_indices_dict[int(test_idx)] = common_candidates[sel].astype(int)

        if return_distances:
            combined_distances_dict[int(test_idx)] = summed_distances[sel].astype(float)

    if return_distances:
        return combined_indices_dict, combined_distances_dict
    return combined_indices_dict

def find_k_closest_frankenstein(
    neighbors_indices_dict_static,
    train_static,
    neighbors_indices_dict_labs,
    train_labs,
    neighbors_indices_dict_meds,
    train_meds,
    neighbors_indices_dict_text,
    train_text,
    indices_test,
    k=5,
):
    """
        Build multimodal "Frankenstein" neighbors from four modality-specific
        nearest-neighbor dictionaries: static, labs, meds, text.

        For each test sample and each rank i=1..k:
          - static/labs/meds use the median over the first i neighbors
            (for i=1 this is the closest neighbor itself)
          - text is copied from the i-th neighbor directly (no interpolation)

        Each output candidate is a dict:
          {
            "tab": <1D static vector>,
            "ts": {"labs": <2D labs>, "meds": <2D meds>},
            "text": <raw text string>,
            "text_input": <raw text object>,
          }

        Neighbor dictionary values can be:
          - integer indices into the corresponding train array, or
          - already-materialized neighbor arrays/values.

        Returns:
            dict[test_idx -> dict[key -> candidate_dict]]
    """

    def _lookup(dct, sample):
        if dct is None:
            return []
        if sample in dct:
            return dct[sample]
        s_int = int(sample)
        return dct.get(s_int, [])

    def _extract_numeric_neighbors(raw_neighbors, train_arr, k_take):
        raw = np.asarray(raw_neighbors)
        is_indices = raw.ndim == 1 and raw.dtype.kind in {"i", "u"}
        if is_indices:
            idx = np.asarray(raw[:k_take], dtype=int)
            vals = np.asarray(train_arr[idx])
            return vals, idx

        vals = np.asarray(raw_neighbors)[:k_take]
        return np.asarray(vals), None

    def _extract_text_neighbors(raw_neighbors, train_text_arr, k_take):
        raw = raw_neighbors

        arr = np.asarray(raw)
        if arr.ndim == 1 and arr.dtype.kind in {"i", "u"}:
            idx = np.asarray(arr[:k_take], dtype=int)
            vals = np.asarray(train_text_arr, dtype=object)[idx]
            return np.asarray(vals, dtype=object), idx

        if isinstance(raw, (list, tuple)) and len(raw) > 0 and isinstance(raw[0], dict):
            vals = np.asarray([item.get("text", "") for item in raw[:k_take]], dtype=object)
            return vals, None

        vals = np.asarray(raw, dtype=object)[:k_take]
        return vals, None

    def _median_or_first(block, use_n):
        block_arr = np.asarray(block)
        if int(use_n) <= 1:
            return np.asarray(block_arr[0], dtype=float)
        return np.asarray(np.median(block_arr[:use_n], axis=0), dtype=float)

    combined_indices_dict = {}
    k = _validate_k(k)
    train_text_arr = np.asarray(train_text, dtype=object).reshape(-1)

    for sample in indices_test:
        raw_static = _lookup(neighbors_indices_dict_static, sample)
        raw_labs = _lookup(neighbors_indices_dict_labs, sample)
        raw_meds = _lookup(neighbors_indices_dict_meds, sample)
        raw_text = _lookup(neighbors_indices_dict_text, sample)

        static_vals, static_idx = _extract_numeric_neighbors(raw_static, train_static, k)
        labs_vals, labs_idx = _extract_numeric_neighbors(raw_labs, train_labs, k)
        meds_vals, meds_idx = _extract_numeric_neighbors(raw_meds, train_meds, k)
        text_vals, text_idx = _extract_text_neighbors(raw_text, train_text_arr, k)

        sample_synthetics = {}
        n_static = int(np.asarray(static_vals).shape[0]) if np.asarray(static_vals).ndim > 0 else 0
        n_labs = int(np.asarray(labs_vals).shape[0]) if np.asarray(labs_vals).ndim > 0 else 0
        n_meds = int(np.asarray(meds_vals).shape[0]) if np.asarray(meds_vals).ndim > 0 else 0
        n_text = int(np.asarray(text_vals).shape[0]) if np.asarray(text_vals).ndim > 0 else 0

        if min(n_static, n_labs, n_meds, n_text) <= 0:
            combined_indices_dict[int(sample)] = sample_synthetics
            continue

        k_eff = min(k, n_static, n_labs, n_meds, n_text)
        for i in range(1, k_eff + 1):
            use_n = i
            static_med = _median_or_first(static_vals, use_n)
            labs_med = _median_or_first(labs_vals, use_n)
            meds_med = _median_or_first(meds_vals, use_n)

            text_anchor = text_vals[i - 1]
            if text_idx is not None:
                text_input = train_text_arr[int(text_idx[i - 1])]
            else:
                text_input = text_anchor

            sample_synthetics[f"k:{i}"] = {
                "tab": static_med,
                "ts": {
                    "labs": labs_med,
                    "meds": meds_med,
                },
                "text": str(text_anchor),
                "text_input": text_input,
                "source_indices": {
                    "static": (None if static_idx is None else int(static_idx[i - 1])),
                    "labs": (None if labs_idx is None else int(labs_idx[i - 1])),
                    "meds": (None if meds_idx is None else int(meds_idx[i - 1])),
                    "text": (None if text_idx is None else int(text_idx[i - 1])),
                },
            }

        combined_indices_dict[int(sample)] = sample_synthetics

    return combined_indices_dict

def find_k_closest_latent_model(
    X_train,
    y_train,
    X_test,
    selected_test_indices,
    model,
    X_train_meds=None,
    X_train_labs=None,
    X_train_text=None,
    X_test_meds=None,
    X_test_labs=None,
    X_test_text=None,
    target_value=0,
    k=5,
    distance_fn=euclidean_distances,
    distance_metric=None,
    return_indices=False,
    precomputed_train_latent=None,
    precomputed_test_latent=None,
    return_latents=False,
):
    """
    Find nearest neighbors in the model latent space.

    Supports:
      - 1-input models: pass X_train and X_test (legacy)
      - 4-input models (meds, labs, static, text): pass
        X_train/X_test as static and provide meds/labs/text arrays through the
        dedicated arguments.

    The latent representation is taken from model.layers[-2].
    Distances use ``distance_metric`` when provided (e.g., "euclidean", "manhattan", "hamming"),
    otherwise ``distance_fn``.
        Returns:
            neighbors_input_dict:
              - 1-input mode: dict[test_idx -> np.ndarray of neighbor inputs]
              - 4-input mode: dict[test_idx -> list[dict]] where each dict has
                keys: 'static','meds','labs','text' plus compatibility keys
                'tab','ts','text_input'
            neighbors_distances_dict: dict mapping each selected test index -> 1D numpy array of corresponding distances (floats)
            If return_indices=True, also returns:
                neighbors_indices_dict: dict mapping each selected test index -> 1D numpy array of neighbor indices (length k_eff)
            If return_latents=True, also returns:
                latent_cache: {"train_latent": ..., "test_latent": ...}
    """
    k = _validate_k(k)
    y_train = np.asarray(y_train).reshape(-1)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    selected_test_indices = _normalize_selected_test_indices(selected_test_indices, X_test.shape[0])

    try:
        n_inputs = len(model.inputs)
    except Exception:
        n_inputs = 1
    use_four_input = (
        int(n_inputs) == 4
        and X_train_meds is not None
        and X_train_labs is not None
        and X_train_text is not None
        and X_test_meds is not None
        and X_test_labs is not None
        and X_test_text is not None
    )

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have matching first dimension.")

    if use_four_input:
        X_train_meds = np.asarray(X_train_meds)
        X_train_labs = np.asarray(X_train_labs)
        X_train_text = np.asarray(X_train_text)
        X_test_meds = np.asarray(X_test_meds)
        X_test_labs = np.asarray(X_test_labs)
        X_test_text = np.asarray(X_test_text)

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        if (
            X_train_meds.shape[0] != n_train
            or X_train_labs.shape[0] != n_train
            or X_train_text.shape[0] != n_train
        ):
            raise ValueError("All train modality arrays must have the same number of patients.")
        if (
            X_test_meds.shape[0] != n_test
            or X_test_labs.shape[0] != n_test
            or X_test_text.shape[0] != n_test
        ):
            raise ValueError("All test modality arrays must have the same number of patients.")

    # Create a new model that excludes the last layer.
    latent_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    # Generate latent representations for the train and test sets (or reuse precomputed).
    if precomputed_train_latent is not None:
        X_train_latent = np.asarray(precomputed_train_latent)
    else:
        if use_four_input:
            X_train_latent = latent_model.predict(
                [X_train_meds, X_train_labs, X_train, X_train_text],
                verbose=0,
            )
        else:
            X_train_latent = latent_model.predict(X_train, verbose=0)

    if precomputed_test_latent is not None:
        X_test_latent = np.asarray(precomputed_test_latent)
    else:
        if use_four_input:
            X_test_latent = latent_model.predict(
                [X_test_meds, X_test_labs, X_test, X_test_text],
                verbose=0,
            )
        else:
            X_test_latent = latent_model.predict(X_test, verbose=0)

    X_train_latent = np.asarray(X_train_latent)
    X_test_latent = np.asarray(X_test_latent)
    if X_train_latent.ndim == 1:
        X_train_latent = X_train_latent.reshape(-1, 1)
    if X_test_latent.ndim == 1:
        X_test_latent = X_test_latent.reshape(-1, 1)
    if X_train_latent.ndim > 2:
        X_train_latent = X_train_latent.reshape(X_train_latent.shape[0], -1)
    if X_test_latent.ndim > 2:
        X_test_latent = X_test_latent.reshape(X_test_latent.shape[0], -1)
    if X_train_latent.shape[0] != X_train.shape[0]:
        raise ValueError("precomputed/train latent rows must match X_train rows.")
    if X_test_latent.shape[0] != X_test.shape[0]:
        raise ValueError("precomputed/test latent rows must match X_test rows.")

    # Candidate indices in X_train_latent that have the requested label.
    candidate_indices = np.flatnonzero(y_train == target_value)
    candidate_samples = X_train_latent[candidate_indices]  # shape (n_candidates, latent_dim)

    n_cand = candidate_samples.shape[0]
    print(f"Number of candidate samples with target value {target_value}: {n_cand}")
    if n_cand == 0:
        empty_neighbors = [] if use_four_input else X_train[:0]
        empty_dist = np.empty((0,), dtype=float)
        neighbors = {int(ti): empty_neighbors for ti in selected_test_indices}
        dists = {int(ti): empty_dist for ti in selected_test_indices}
        if return_indices:
            empty_idx = np.empty((0,), dtype=int)
            return neighbors, dists, {int(ti): empty_idx for ti in selected_test_indices}
        return neighbors, dists

    k_eff = min(k, n_cand)
    neighbors_input_dict = {}
    neighbors_distances_dict = {}
    neighbors_indices_dict = {} if return_indices else None

    sel_test = np.asarray(selected_test_indices, dtype=int)
    query_samples = X_test_latent[sel_test]

    dist_matrix = _pairwise_vector_distances(
        query_samples,
        candidate_samples,
        distance_fn=distance_fn,
        distance_metric=distance_metric,
        default_metric="euclidean",
    )

    for r, test_idx in enumerate(sel_test):
        dists = dist_matrix[r]
        # Get k_eff smallest distances (unsorted via argpartition then sorted).
        if k_eff == n_cand:
            sel_local = np.argsort(dists)[:k_eff]
        else:
            kth = max(0, min(k_eff - 1, dists.shape[0] - 1))
            idx_k = np.argpartition(dists, kth)[:k_eff]
            sel_local = idx_k[np.argsort(dists[idx_k])]

        selected_train_indices = candidate_indices[sel_local].astype(int)
        if use_four_input:
            neighbors_input_dict[int(test_idx)] = [
                {
                    "static": np.asarray(X_train[s_idx]),
                    "meds": np.asarray(X_train_meds[s_idx]),
                    "labs": np.asarray(X_train_labs[s_idx]),
                    "text": str(X_train_text[s_idx]),
                    "tab": np.asarray(X_train[s_idx]),
                    "ts": {
                        "labs": np.asarray(X_train_labs[s_idx]),
                        "meds": np.asarray(X_train_meds[s_idx]),
                    },
                    "text_input": X_train_text[s_idx],
                }
                for s_idx in selected_train_indices
            ]
        else:
            neighbors_input_dict[int(test_idx)] = X_train[selected_train_indices]
        neighbors_distances_dict[int(test_idx)] = dists[sel_local].astype(float)
        if return_indices:
            neighbors_indices_dict[int(test_idx)] = selected_train_indices

    latent_cache = {"train_latent": X_train_latent, "test_latent": X_test_latent}
    if return_indices and return_latents:
        return neighbors_input_dict, neighbors_distances_dict, neighbors_indices_dict, latent_cache
    if return_indices:
        return neighbors_input_dict, neighbors_distances_dict, neighbors_indices_dict
    if return_latents:
        return neighbors_input_dict, neighbors_distances_dict, latent_cache
    return neighbors_input_dict, neighbors_distances_dict


###################################################
# Misc
###################################################

def build_concat_dict(indices, x_static_hat, x_labs_hat, x_meds_hat=None, x_text_hat=None):
    """
    Build a dict: {index -> [numpy_array, ...]} where each array is an early-fusion
    sample with shape (T, D_total).

    New 4-modality layout (preferred):
      output features are concatenated as [static, labs, meds, text]
      with static/text repeated across time.

    Legacy 2-modality layout is preserved when only (x_static_hat, x_labs_hat)
    are provided, where x_labs_hat is interpreted as the old time-series branch.
    """
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise TypeError("indices must be a list/tuple/np.ndarray of indices")

    n = len(indices)
    if n == 0:
        return {}

    def _to_np(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    static = _to_np(x_static_hat)
    labs = _to_np(x_labs_hat)
    meds = None if x_meds_hat is None else _to_np(x_meds_hat)
    text = None if x_text_hat is None else _to_np(x_text_hat)

    legacy_mode = meds is None and text is None

    def _concat_legacy(static_vec, ts_mat):
        static_vec = np.asarray(static_vec)
        ts_mat = np.asarray(ts_mat)
        if static_vec.ndim != 1 or ts_mat.ndim != 2:
            raise ValueError(f"Legacy mode expects static 1D and ts 2D. Got {static_vec.shape}, {ts_mat.shape}")
        t = ts_mat.shape[0]
        static_rep = np.broadcast_to(static_vec[None, :], (t, static_vec.shape[0]))
        return np.concatenate([static_rep, ts_mat], axis=1)

    def _concat_4mod(static_vec, labs_mat, meds_mat, text_sample):
        static_vec = np.asarray(static_vec)
        labs_mat = np.asarray(labs_mat)
        meds_mat = np.asarray(meds_mat)
        text_sample = np.asarray(text_sample)

        if static_vec.ndim != 1:
            raise ValueError("static sample must be 1D")
        if labs_mat.ndim != 2 or meds_mat.ndim != 2:
            raise ValueError("labs/meds samples must be 2D")
        if labs_mat.shape[0] != meds_mat.shape[0]:
            raise ValueError("labs and meds must have same time dimension")

        # text may be provided either as a vector [D_text] or a sequence [T_text, D_text].
        if text_sample.ndim == 2:
            text_vec = text_sample.mean(axis=0)
        elif text_sample.ndim == 1:
            text_vec = text_sample
        else:
            raise ValueError(f"text sample must be 1D or 2D. Got shape={text_sample.shape}")

        t = labs_mat.shape[0]
        static_rep = np.broadcast_to(static_vec[None, :], (t, static_vec.shape[0]))
        text_rep = np.broadcast_to(text_vec[None, :], (t, text_vec.shape[0]))
        return np.concatenate([static_rep, labs_mat, meds_mat, text_rep], axis=1)

    out = {}

    # Grouped layout
    if static.ndim == 3 and labs.ndim == 4:
        if static.shape[0] != n or labs.shape[0] != n:
            raise ValueError("Grouped layout expects first dim to match len(indices)")

        for i, idx in enumerate(indices):
            samples = []
            for j in range(static.shape[1]):
                if legacy_mode:
                    arr = _concat_legacy(static[i, j], labs[i, j])
                else:
                    arr = _concat_4mod(static[i, j], labs[i, j], meds[i, j], text[i, j])
                samples.append(arr)
            out[int(idx)] = samples
        return out

    # Flat layout
    if static.ndim != 2 or labs.ndim != 3:
        raise ValueError(
            f"Unsupported shapes: static.ndim={static.ndim}, labs.ndim={labs.ndim}."
        )

    if static.shape[0] != labs.shape[0]:
        raise ValueError("Mismatched number of generated samples between static and labs/ts")

    if static.shape[0] % n != 0:
        raise ValueError(
            f"Total generated samples ({static.shape[0]}) must be divisible by len(indices) ({n})."
        )

    n_samples = static.shape[0] // n

    for i, idx in enumerate(indices):
        start = i * n_samples
        end = start + n_samples
        samples = []
        for k in range(start, end):
            if legacy_mode:
                arr = _concat_legacy(static[k], labs[k])
            else:
                arr = _concat_4mod(static[k], labs[k], meds[k], text[k])
            samples.append(arr)
        out[int(idx)] = samples

    return out

###################################################
# Reconstruction helpers
###################################################
def _to_series(pred_dict):
    """Convert prediction dict to pd.Series for easier DataFrame construction."""
    return pd.Series({k: v for k, v in pred_dict.items()})


def _ef_from_static_neighbors(static_neighbors, ts_reference):
    static_neighbors = np.asarray(static_neighbors)
    ts_reference = np.asarray(ts_reference)

    if static_neighbors.ndim != 2:
        raise ValueError(f"static_neighbors must be (B, D_static). Got {static_neighbors.shape}")
    if ts_reference.ndim != 2:
        raise ValueError(f"ts_reference must be (T, D_ts). Got {ts_reference.shape}")

    b = static_neighbors.shape[0]
    t = ts_reference.shape[0]

    static_rep = np.repeat(static_neighbors[:, None, :], t, axis=1)
    ts_rep = np.repeat(ts_reference[None, :, :], b, axis=0)
    return np.concatenate((static_rep, ts_rep), axis=2)


def _ef_from_ts_neighbors(ts_neighbors, static_reference):
    ts_neighbors = np.asarray(ts_neighbors)
    static_reference = np.asarray(static_reference)

    if ts_neighbors.ndim != 3:
        raise ValueError(f"ts_neighbors must be (B, T, D_ts). Got {ts_neighbors.shape}")
    if static_reference.ndim != 1:
        raise ValueError(f"static_reference must be (D_static,). Got {static_reference.shape}")

    b, t, _ = ts_neighbors.shape
    static_batch = np.repeat(static_reference[None, :], b, axis=0)
    static_rep = np.repeat(static_batch[:, None, :], t, axis=1)
    return np.concatenate((static_rep, ts_neighbors), axis=2)


def _ef_from_compressed_neighbors(tabular_neighbors, timesteps, n_static):
    tabular_neighbors = np.asarray(tabular_neighbors)
    if tabular_neighbors.ndim != 2:
        raise ValueError(f"tabular_neighbors must be (B, D_static+D_ts). Got {tabular_neighbors.shape}")

    static_part = tabular_neighbors[:, :n_static]
    ts_mean = tabular_neighbors[:, n_static:]

    static_rep = np.repeat(static_part[:, None, :], timesteps, axis=1)
    ts_rep = np.repeat(ts_mean[:, None, :], timesteps, axis=1)
    return np.concatenate((static_rep, ts_rep), axis=2)


def _ef_from_combined_parts(parts):
    static_part = np.asarray(parts["static"])
    ts_part = np.asarray(parts["ts"])

    if static_part.ndim != 2:
        raise ValueError(f"parts['static'] must be (B, D_static). Got {static_part.shape}")
    if ts_part.ndim != 3:
        raise ValueError(f"parts['ts'] must be (B, T, D_ts). Got {ts_part.shape}")

    b, t, _ = ts_part.shape
    static_rep = np.repeat(static_part[:, None, :], t, axis=1)
    return np.concatenate((static_rep, ts_part), axis=2)


def reconstruct_genetic_counterfactuals(
    solutions,
    index,
    tab_features,
    test_ts,
    ts_features,
    segments,
    tau_c,
    K=None,
):
    """Reconstruct counterfactuals from genetic (pymoo) solutions.

    Supports two formats for `solutions[index]`:
    1) Raw `pymoo.Result` with attribute `.X` (older runs)
    2) Lightweight dict payload produced by `_run_patient` with key "X" (recommended)
    """
    if K is None:
        K = globals().get("K", None)

    entry = solutions[index]

    # If the worker returned a structured payload, validate it.
    if isinstance(entry, dict):
        if not entry.get("success", True):
            raise RuntimeError(
                f"Patient {index} failed during generation: {entry.get('error')}.\n{entry.get('traceback', '')}"
            )
        selected_X = entry.get("X", None)
        payload_k_labs = entry.get("K_labs", None)
        payload_k_meds = entry.get("K_meds", None)
        payload_k_text = entry.get("K_text", 0)
        payload_text_candidates = entry.get("text_candidates", None)
        payload_text_input_candidates = entry.get("text_input_candidates", None)
    else:
        selected_X = getattr(entry, "X", None)
        payload_k_labs = None
        payload_k_meds = None
        payload_text_candidates = None
        payload_text_input_candidates = None

    if selected_X is None:
        raise ValueError(f"No solutions found for patient {index} (X is None).")

    counterfactuals = {}
    tab_cf = {}
    ts_cf = {}

    selected_solutions = np.asarray(selected_X)  # (n_solutions, n_vars)
    split_mode = isinstance(test_ts, dict) and isinstance(tau_c, dict) and isinstance(segments, dict)
    if split_mode:
        for key in ("labs", "meds"):
            if key not in test_ts or key not in tau_c or key not in segments:
                raise ValueError("Split mode expects test_ts/tau_c/segments dicts with keys {'labs','meds'}.")
    else:
        if K is None:
            raise ValueError("K must be provided or defined globally for non-split TS reconstruction.")
        test_ts_packed = np.asarray(test_ts)
        tau_c_packed = np.asarray(tau_c)

    for i in range(selected_solutions.shape[0]):
        x = selected_solutions[i]

        x_tab_cf = x[:tab_features]

        rem = x.size - tab_features
        if rem < 0:
            raise ValueError(f"tab_features={tab_features} is larger than solution length={x.size}.")
        if split_mode:
            k_text = int(payload_k_text) if payload_k_text is not None else 0
            ts_rem = rem - (k_text * TEXT_EDIT_SLOT_DIM)
            if ts_rem < 0:
                raise ValueError("Remaining decision vector size is negative after accounting for text edits.")

            if ts_rem % EDIT_SLOT_DIM != 0:
                raise ValueError(
                    f"Time-series edit part length {ts_rem} is not divisible by EDIT_SLOT_DIM={EDIT_SLOT_DIM}."
                )

            k_total = ts_rem // EDIT_SLOT_DIM
            k_labs = int(payload_k_labs) if payload_k_labs is not None else None
            k_meds = int(payload_k_meds) if payload_k_meds is not None else None

            if k_labs is None or k_meds is None:
                if K is not None and int(2 * K) == int(k_total):
                    k_labs = int(K)
                    k_meds = int(K)
                else:
                    k_labs = int(k_total // 2)
                    k_meds = int(k_total - k_labs)

            if int(k_labs) + int(k_meds) != int(k_total):
                raise ValueError(
                    f"Inconsistent slot split: k_labs={k_labs}, k_meds={k_meds}, total={k_total}."
                )

            off = tab_features
            X_labs_edits = x[off : off + k_labs * EDIT_SLOT_DIM].reshape(k_labs, EDIT_SLOT_DIM)
            off += k_labs * EDIT_SLOT_DIM
            X_meds_edits = x[off : off + k_meds * EDIT_SLOT_DIM].reshape(k_meds, EDIT_SLOT_DIM)

            decoded_labs = decode_ts_edits(
                X_labs_edits,
                n_segments=len(segments["labs"]),
                n_channels=int(np.asarray(test_ts["labs"][index]).shape[1]),
                tau_c=np.asarray(tau_c["labs"]).reshape(-1),
            )
            decoded_meds = decode_ts_edits(
                X_meds_edits,
                n_segments=len(segments["meds"]),
                n_channels=int(np.asarray(test_ts["meds"][index]).shape[1]),
                tau_c=np.asarray(tau_c["meds"]).reshape(-1),
            )
            x_labs_cf = apply_decoded_edits(np.asarray(test_ts["labs"][index]), decoded_labs, segments["labs"])
            x_meds_cf = apply_decoded_edits(np.asarray(test_ts["meds"][index]), decoded_meds, segments["meds"])
            x_ts_cf = {"labs": x_labs_cf, "meds": x_meds_cf}
            cf_item = {"tab": np.asarray(x_tab_cf), "ts": x_ts_cf}
            if payload_text_candidates is not None and i < len(payload_text_candidates):
                cf_item["text"] = payload_text_candidates[i]
            if payload_text_input_candidates is not None and i < len(payload_text_input_candidates):
                cf_item["text_input"] = payload_text_input_candidates[i]
            counterfactuals[i] = cf_item
            tab_cf[i] = np.asarray(x_tab_cf)
            ts_cf[i] = x_ts_cf
        else:
            k_text = int(payload_k_text) if payload_k_text is not None else 0
            ts_rem = rem - (k_text * TEXT_EDIT_SLOT_DIM)
            if ts_rem < 0:
                raise ValueError("Remaining decision vector size is negative after accounting for text edits.")

            if K > 0 and ts_rem % K != 0:
                raise ValueError(
                    f"Time-series edit part length {ts_rem} is not divisible by K={K}. "
                    f"Cannot reshape edits to (K, -1)."
                )
            edit_dims = ts_rem // K if K > 0 else 0
            X_edits = x[tab_features : tab_features + ts_rem].reshape(K, edit_dims)

            decoded = decode_ts_edits(
                X_edits,
                n_segments=len(segments),
                n_channels=ts_features,
                tau_c=tau_c_packed,
            )

            x_ts_cf = apply_decoded_edits(test_ts_packed[index], decoded, segments)

            t = x_ts_cf.shape[0]
            x_tab_rep = np.broadcast_to(np.asarray(x_tab_cf)[None, :], (t, tab_features))
            x_cf = np.concatenate([x_tab_rep, np.asarray(x_ts_cf)], axis=1)
            x_cf = np.asarray(x_cf, dtype=np.float32)[None, :, :]

            counterfactuals[i] = x_cf
            tab_cf[i] = x_tab_cf
            ts_cf[i] = x_ts_cf

    return counterfactuals, tab_cf, ts_cf


###################################################
# Prediction helpers
###################################################

def predict_for_samples(samples, batch_size=32):
    """Predict for a batch of already-materialized samples.

    Parameters
    ----------
    samples:
        - For 1-input models: array shaped (T, D) or (B, T, D)
        - For 2-input models: list/tuple [static, ts] where
          static is (B, D_static) and ts is (B, T, D_ts)

    Returns
    -------
    dict
        {neighbor_rank: prediction}
    """

    # determine number of model inputs safely
    try:
        n_inputs = len(trained_model.inputs)
    except Exception:
        n_inputs = 1

    def _ensure_batch_3d(a):
        a = np.asarray(a)
        if a.ndim == 2:
            a = a[None, :, :]
        elif a.ndim == 3:
            pass
        else:
            raise ValueError(f"Expected (T, D) or (B, T, D). Got shape={a.shape}")
        return a

    if n_inputs == 1:
        x = _ensure_batch_3d(samples)
        inputs = x
        batch_len = x.shape[0]
    else:
        if not isinstance(samples, (list, tuple)) or len(samples) != 2:
            raise ValueError("For multi-input models, pass samples as [static, ts]")
        static, ts = samples
        static = np.asarray(static)
        ts = _ensure_batch_3d(ts)
        if static.ndim == 1:
            static = static[None, :]
        inputs = [static, ts]
        batch_len = ts.shape[0]

    # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(True)
    try:
        trained_model.run_eagerly = True
    except Exception:
        pass

    with tf.device("/CPU:0"):
        preds = trained_model.predict(inputs, batch_size=batch_size)

    if isinstance(preds, list):
        return {int(i): [p[i] for p in preds] for i in range(batch_len)}
    return {int(i): preds[i] for i in range(batch_len)}


def predict_for_indices(selected_indices, batch_size=32):
    """
    Run the loaded `trained_model` on the samples indexed by `selected_indices` and
    return a dict mapping each index -> model prediction (numpy array).
    Works for single-input models (expects X_test) and two-input models
    (expects test_static and test_ts).
    """

    idx = np.asarray(selected_indices, dtype=int)
    # determine number of model inputs safely
    try:
        n_inputs = len(trained_model.inputs)
    except Exception:
        n_inputs = 1

    if n_inputs == 1:
        inputs = X_train[idx]
    else:
        inputs = [train_static[idx], train_ts[idx]]

    # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(True)
    try:
        trained_model.run_eagerly = True
    except Exception:
        pass

    with tf.device("/CPU:0"):
        preds = trained_model.predict(inputs, batch_size=batch_size)
    # preds shape: (n_samples, ...) or list if multiple outputs; normalize to list
    if isinstance(preds, list):
        # multiple outputs: map each index to list of output arrays
        return {int(i): [p[j] for p in preds] for j, i in enumerate(idx)}
    else:
        return {int(i): preds[j] for j, i in enumerate(idx)}


def predict_for_frankenstein(frankenstein_neighbors, batch_size=32):
    """
    Run the loaded `trained_model` on the frankenstein neighbors and
    return a dict mapping each test-sample -> dict of synthetic-key -> prediction(s).
    Expected input format: { sample_idx: { synth_key: array_or_list, ... }, ... }
    For each synthetic array we return a single prediction (1-D). If a synthetic
    value is a list of arrays we predict each element individually and return a list
    of predictions for that synthetic key.
    """

    preds_frankenstein = {}
    for sample_key, synthetics in frankenstein_neighbors.items():
        # Normalize sample key to int when possible
        out_key = int(sample_key) if str(sample_key).isdigit() else sample_key
        preds_for_sample = {}
        # synthetics is expected to be a dict mapping synth_key -> array or list
        if isinstance(synthetics, dict):
            for synth_key, arr in synthetics.items():
                # If arr is a list/tuple of per-synthetic arrays, predict each separately
                if isinstance(arr, (list, tuple)):
                    preds_list = []
                    for sample in arr:
                        # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
                        tf.config.optimizer.set_jit(False)
                        tf.config.run_functions_eagerly(True)
                        try:
                            trained_model.run_eagerly = True
                        except Exception:
                            pass

                        with tf.device("/CPU:0"):
                            p = trained_model.predict(sample, batch_size=1)
                        if isinstance(p, list):
                            preds_list.append([out.reshape(-1) for out in p])
                        else:
                            preds_list.append(p.reshape(-1))
                    preds_for_sample[synth_key] = preds_list
                else:
                    # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
                    tf.config.optimizer.set_jit(False)
                    tf.config.run_functions_eagerly(True)
                    try:
                        trained_model.run_eagerly = True
                    except Exception:
                        pass

                    with tf.device("/CPU:0"):
                        p = trained_model.predict(arr, batch_size=1)
                    if isinstance(p, list):
                        preds_for_sample[synth_key] = [out.reshape(-1) for out in p]
                    else:
                        preds_for_sample[synth_key] = p.reshape(-1)
        else:
            # Backwards compatibility: if top-level mapping was flat, predict as before
            arr = synthetics
            if isinstance(arr, (list, tuple)):
                preds_list = []
                for sample in arr:
                    # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
                    tf.config.optimizer.set_jit(False)
                    tf.config.run_functions_eagerly(True)
                    try:
                        trained_model.run_eagerly = True
                    except Exception:
                        pass

                    with tf.device("/CPU:0"):
                        p = trained_model.predict(sample, batch_size=1)
                    if isinstance(p, list):
                        preds_list.append([out.reshape(-1) for out in p])
                    else:
                        preds_list.append(p.reshape(-1))
                preds_for_sample = preds_list
            else:
                # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
                tf.config.optimizer.set_jit(False)
                tf.config.run_functions_eagerly(True)
                try:
                    trained_model.run_eagerly = True
                except Exception:
                    pass

                with tf.device("/CPU:0"):
                    p = trained_model.predict(arr, batch_size=batch_size)
                if isinstance(p, list):
                    preds_for_sample = [out.reshape(-1) for out in p]
                else:
                    preds_for_sample = p.reshape(-1)
        preds_frankenstein[out_key] = preds_for_sample
    return preds_frankenstein


def predict_for_VAE(counterfactuals_dict, batch_size=32, trained_model=None, vae_device=None):
    """
    counterfactuals_dict: { sample_idx: [list_of_counterfactuals], ... }
    list_of_counterfactuals can contain:
      - legacy arrays (single-input models), or
      - dict-style multimodal CFs with static/labs/meds/text parts.

    For 4-input models, prediction is called uniformly as:
      [x_meds_b, x_labs_b, x_static_b, x_text_b]

    Returns
    -------
    preds_vae : dict
        { sample_idx: [pred_for_cf0, pred_for_cf1, ...], ... }
    batched_counterfactuals : dict
        { sample_idx: [batched_input_for_cf0, ...], ... }
        where batched input is either:
          - np.ndarray (legacy single-input), or
          - list [x_meds_b, x_labs_b, x_static_b, x_text_b] for 4-input models
    """

    def _to_numpy(a):
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    def _move_model_to_device(model_obj, device: str):
        """Best-effort move for Keras(torch-backend) / torch modules."""
        moved = False
        try:
            if hasattr(model_obj, "to"):
                model_obj.to(device)
                moved = True
        except Exception:
            pass
        try:
            torch_module = getattr(model_obj, "_torch_module", None)
            if torch_module is not None and hasattr(torch_module, "to"):
                torch_module.to(device)
                moved = True
        except Exception:
            pass
        return moved

    def _predict_safe(model_obj, model_in, bsz, preferred_device):
        """Predict with automatic recovery from CUDA device-mismatch errors."""
        try:
            return model_obj.predict(model_in, batch_size=bsz, verbose=0)
        except Exception as e:
            msg = str(e)
            if "Expected all tensors to be on the same device" not in msg:
                raise
            # First, enforce chosen single device; if still failing, fall back to CPU.
            _move_model_to_device(model_obj, preferred_device)
            try:
                return model_obj.predict(model_in, batch_size=bsz, verbose=0)
            except Exception:
                _move_model_to_device(model_obj, "cpu")
                return model_obj.predict(model_in, batch_size=bsz, verbose=0)

    def _ensure_batch_3d(a):
        a = _to_numpy(a).astype(np.float32, copy=False)

        # If a is (T, D) -> make it (1, T, D)
        if a.ndim == 2:
            a = a[None, :, :]
        # If already (B, T, D) keep as is
        elif a.ndim == 3:
            pass
        else:
            raise ValueError(
                f"Each counterfactual must have shape (T, D) or (B, T, D). Got shape={a.shape}"
            )

        return a

    def _infer_text_dim_from_model(m):
        try:
            d = m.inputs[3].shape[-1]
            if d is None:
                return 1
            return int(d)
        except Exception:
            return 1

    def _build_text_embedding_adapter(model_obj):
        """
        Build an inference model that bypasses SweDeClinBert in the text branch and
        accepts text embeddings directly as the 4th input.
        """
        try:
            if len(model_obj.inputs) != 4:
                return model_obj, False
        except Exception:
            return model_obj, False

        concat_idx = None
        for i, lyr in enumerate(model_obj.layers):
            if isinstance(lyr, keras.layers.Concatenate):
                concat_idx = i
                break
        if concat_idx is None:
            return model_obj, False

        branch_layers = [
            lyr for lyr in model_obj.layers[:concat_idx]
            if isinstance(lyr, keras.Model) and not isinstance(lyr, keras.Sequential)
        ]
        if len(branch_layers) < 4:
            return model_obj, False

        text_branch = None
        for b in branch_layers:
            nested = getattr(b, "layers", [])
            if any(
                (nl.__class__.__name__ == "SweDeClinBert") or ("language_model" in nl.name.lower())
                for nl in nested
            ):
                text_branch = b
                break
        if text_branch is None:
            return model_obj, False

        nested = list(getattr(text_branch, "layers", []))
        lm_idx = None
        for i, nl in enumerate(nested):
            if (nl.__class__.__name__ == "SweDeClinBert") or ("language_model" in nl.name.lower()):
                lm_idx = i
                break
        if lm_idx is None:
            return model_obj, False

        post_layers = [nl for nl in nested[lm_idx + 1:] if not isinstance(nl, keras.layers.InputLayer)]
        if len(post_layers) == 0:
            return model_obj, False

        def _shape_wo_batch(t):
            if isinstance(t, (list, tuple)):
                if len(t) == 0:
                    return None
                t = t[0]
            shp = getattr(t, "shape", None)
            if shp is None:
                return None
            return tuple(int(d) if d is not None else None for d in shp[1:])

        first_post = post_layers[0]
        emb_shape = _shape_wo_batch(getattr(first_post, "input", None))
        if emb_shape is None:
            return model_obj, False
        text_emb_input = keras.layers.Input(shape=emb_shape, name=f"{text_branch.name}_emb_input")
        z = text_emb_input
        for pl in post_layers:
            z = pl(z)
        text_branch_no_bert = keras.Model(text_emb_input, z, name=f"{text_branch.name}_no_bert")

        # Map each branch layer to its original input index by matching input shapes.
        branch_to_input = {}
        used = set()
        for b in branch_layers:
            b_shape = _shape_wo_batch(getattr(b, "input", None))
            if b_shape is None:
                continue
            matched = None
            for j, inp in enumerate(model_obj.inputs):
                if j in used:
                    continue
                in_shape = _shape_wo_batch(inp)
                if in_shape is None:
                    continue
                if in_shape == b_shape:
                    matched = j
                    break
            if matched is None and b is text_branch:
                matched = 3
            if matched is not None:
                branch_to_input[b.name] = matched
                used.add(matched)

        text_input_idx = int(branch_to_input.get(text_branch.name, 3))

        new_inputs = []
        for j, inp in enumerate(model_obj.inputs):
            if j == text_input_idx:
                new_inputs.append(text_emb_input)
            else:
                in_shape = tuple(int(d) if d is not None else None for d in inp.shape[1:])
                new_inputs.append(keras.layers.Input(shape=in_shape, name=f"{inp.name.split(':')[0]}_adapted"))

        branch_outputs = {}
        for b in branch_layers:
            in_idx = branch_to_input.get(b.name, None)
            if in_idx is None:
                continue
            if b is text_branch:
                branch_outputs[b.name] = text_branch_no_bert(new_inputs[in_idx])
            else:
                branch_outputs[b.name] = b(new_inputs[in_idx])

        concat_layer = model_obj.layers[concat_idx]
        concat_inputs = []
        for t in concat_layer.input:
            src = t._keras_history[0]
            if src.name not in branch_outputs:
                return model_obj, False
            concat_inputs.append(branch_outputs[src.name])

        x = concat_layer(concat_inputs)
        for lyr in model_obj.layers[concat_idx + 1:]:
            x = lyr(x)

        adapted_model = keras.Model(new_inputs, x, name=f"{model_obj.name}_text_emb")
        return adapted_model, True

    def _safe_text_vec(text_val, text_dim_hint=1):
        t = np.asarray(text_val)
        if t.ndim == 0:
            try:
                v = np.array([float(t)], dtype=np.float32)
            except Exception:
                v = np.zeros((1,), dtype=np.float32)
        elif t.ndim == 1:
            try:
                v = t.astype(np.float32, copy=False)
            except Exception:
                v = np.zeros((1,), dtype=np.float32)
        elif t.ndim == 2:
            try:
                v = t.astype(np.float32, copy=False).mean(axis=0)
            except Exception:
                v = np.zeros((1,), dtype=np.float32)
        else:
            v = np.zeros((1,), dtype=np.float32)

        v = v.reshape(-1)
        d = int(max(1, text_dim_hint))
        if v.size == d:
            return v
        if v.size > d:
            return v[:d]
        out = np.zeros((d,), dtype=np.float32)
        out[: v.size] = v
        return out

    def _cf_to_4input_batch(cf):
        if not isinstance(cf, dict):
            raise TypeError(
                "For 4-input models, each counterfactual must be a dict with static/labs/meds/text parts."
            )

        if "static" in cf:
            x_static = np.asarray(cf["static"], dtype=np.float32).reshape(1, -1)
        elif "tab" in cf:
            x_static = np.asarray(cf["tab"], dtype=np.float32).reshape(1, -1)
        else:
            raise ValueError("Counterfactual dict missing 'static'/'tab' key.")

        if "labs" in cf and "meds" in cf:
            x_labs = np.asarray(cf["labs"], dtype=np.float32)
            x_meds = np.asarray(cf["meds"], dtype=np.float32)
        elif "ts" in cf and isinstance(cf["ts"], dict):
            x_labs = np.asarray(cf["ts"].get("labs"), dtype=np.float32)
            x_meds = np.asarray(cf["ts"].get("meds"), dtype=np.float32)
        else:
            raise ValueError("Counterfactual dict missing labs/meds information.")

        if x_labs.ndim != 2 or x_meds.ndim != 2:
            raise ValueError(
                f"Expected labs/meds as 2D arrays, got labs={x_labs.shape}, meds={x_meds.shape}"
            )

        x_labs_b = x_labs[None, :, :]
        x_meds_b = x_meds[None, :, :]

        text_src = cf.get("text_input", cf.get("text", 0.0))
        try:
            text_rank = len(model_for_pred.inputs[3].shape)
        except Exception:
            text_rank = 3

        if use_text_embedding_input:
            text_arr = np.asarray(text_src, dtype=np.float32)
            if text_arr.ndim == 0:
                text_arr = text_arr.reshape(1)
            if int(text_rank) == 3:
                # Expected text shape: (B, T, D)
                if text_arr.ndim == 1:
                    x_text_b = text_arr[None, None, :]
                elif text_arr.ndim == 2:
                    x_text_b = text_arr[None, :, :]
                elif text_arr.ndim == 3 and text_arr.shape[0] == 1:
                    x_text_b = text_arr
                else:
                    raise ValueError(f"Unexpected text embedding shape for rank-3 input: {text_arr.shape}")
            else:
                # Expected text shape: (B, D)
                if text_arr.ndim == 1:
                    x_text_b = text_arr[None, :]
                elif text_arr.ndim == 2:
                    # Sequence -> pooled vector.
                    x_text_b = text_arr.mean(axis=0, keepdims=True)
                elif text_arr.ndim == 3 and text_arr.shape[0] == 1:
                    x_text_b = text_arr.mean(axis=1)
                else:
                    raise ValueError(f"Unexpected text embedding shape for rank-2 input: {text_arr.shape}")
        else:
            # Legacy PID/text path (model still contains SweDeClinBert input path).
            text_dim = _infer_text_dim_from_model(model_for_pred)
            text_vec = _safe_text_vec(text_src, text_dim_hint=text_dim)
            if int(text_rank) == 2:
                x_text_b = text_vec[None, :].astype(np.float32, copy=False)
            else:
                x_text_b = np.broadcast_to(
                    text_vec[None, None, :], (1, x_labs_b.shape[1], text_vec.shape[0])
                ).astype(np.float32, copy=False)

        return [x_meds_b, x_labs_b, x_static.astype(np.float32, copy=False), x_text_b]

    def _explode_batched_cf(item):
        """Turn one potentially batched CF item into a list of per-candidate CF items."""
        if not isinstance(item, dict):
            return [item]

        # Detect batched multimodal dicts: labs/meds as (N, T, D), static as (N, D).
        labs_src = item.get("labs", item.get("ts", {}).get("labs") if isinstance(item.get("ts"), dict) else None)
        meds_src = item.get("meds", item.get("ts", {}).get("meds") if isinstance(item.get("ts"), dict) else None)
        static_src = item.get("static", item.get("tab", None))
        text_src = item.get("text_input", item.get("text", None))

        labs_arr = None if labs_src is None else np.asarray(labs_src)
        meds_arr = None if meds_src is None else np.asarray(meds_src)
        static_arr = None if static_src is None else np.asarray(static_src)

        if (
            labs_arr is not None and meds_arr is not None and static_arr is not None
            and labs_arr.ndim == 3 and meds_arr.ndim == 3 and static_arr.ndim == 2
        ):
            n = int(labs_arr.shape[0])
            if meds_arr.shape[0] != n or static_arr.shape[0] != n:
                raise ValueError(
                    "Batched CF dict has inconsistent candidate dimension between static/labs/meds."
                )

            # Normalize text source to per-candidate list.
            if isinstance(text_src, np.ndarray):
                if text_src.ndim >= 1 and text_src.shape[0] == n:
                    text_list = [text_src[i] for i in range(n)]
                else:
                    text_list = [text_src] * n
            elif isinstance(text_src, (list, tuple)) and len(text_src) == n:
                text_list = list(text_src)
            elif isinstance(text_src, (list, tuple)) and len(np.asarray(text_src, dtype=object).reshape(-1)) == n:
                text_list = list(np.asarray(text_src, dtype=object).reshape(-1))
            else:
                text_list = [text_src] * n

            out = []
            for i in range(n):
                out.append(
                    {
                        "static": static_arr[i],
                        "labs": labs_arr[i],
                        "meds": meds_arr[i],
                        "text_input": text_list[i],
                        "text": (None if text_list[i] is None else str(text_list[i])),
                        "tab": static_arr[i],
                        "ts": {"labs": labs_arr[i], "meds": meds_arr[i]},
                    }
                )
            return out

        return [item]

    # Avoid Keras/TF graph + XLA/JIT issues during predict() by forcing eager + CPU execution.
    tf.config.optimizer.set_jit(False)
    tf.config.run_functions_eagerly(True)
    try:
        trained_model.run_eagerly = True
    except Exception:
        pass

    # Pin all VAE predictions to one device to avoid cross-GPU tensor mixes.
    if vae_device is None:
        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        if isinstance(vae_device, int):
            target_device = f"cuda:{int(vae_device)}"
        else:
            target_device = str(vae_device)
    if target_device.startswith("cuda"):
        try:
            gpu_idx = int(target_device.split(":")[1]) if ":" in target_device else 0
            torch.cuda.set_device(gpu_idx)
        except Exception:
            target_device = "cpu"
    _move_model_to_device(trained_model, target_device)

    try:
        n_model_inputs = len(trained_model.inputs)
    except Exception:
        n_model_inputs = 1

    model_for_pred = trained_model
    use_text_embedding_input = False
    if int(n_model_inputs) == 4:
        model_for_pred, use_text_embedding_input = _build_text_embedding_adapter(trained_model)
        if use_text_embedding_input:
            _move_model_to_device(model_for_pred, target_device)
            try:
                model_for_pred.run_eagerly = True
            except Exception:
                pass

    preds_vae = {}
    batched_counterfactuals = {}

    for sample_key, arr_list in counterfactuals_dict.items():
        out_key = int(sample_key) if str(sample_key).isdigit() else sample_key

        # Be tolerant if a single counterfactual is provided instead of a list.
        if not isinstance(arr_list, (list, tuple)):
            arr_list = [arr_list]
        # Also split dict items that contain a whole batch of candidates.
        expanded = []
        for item in arr_list:
            expanded.extend(_explode_batched_cf(item))
        arr_list = expanded

        preds_list = []
        batched_list = []

        for arr in arr_list:
            if int(n_model_inputs) == 4:
                model_in = _cf_to_4input_batch(arr)
            else:
                model_in = _ensure_batch_3d(arr)  # (1, T, D)
            batched_list.append(model_in)

            p = _predict_safe(model_for_pred, model_in, batch_size, target_device)

            if isinstance(p, list):
                preds_list.append([out.reshape(-1) for out in p])
            else:
                preds_list.append(p.reshape(-1))

        preds_vae[out_key] = preds_list
        batched_counterfactuals[out_key] = batched_list

    return preds_vae, batched_counterfactuals


def create_pred_df(
    neighbors_dict_static,
    neighbors_dict_ts,
    neighbors_dict_EF,
    neighbors_dict_EF_compressed,
    neighbors_dict_frankenstein,
    neighbors_dict_IF,
    neighbors_dict_combined=None,
    index=None,
):
    n_static = int(train_static.shape[1])
    timesteps = int(test_ts[index].shape[0])

    # Build early-fusion inputs from the neighbor parts
    x_EF = np.asarray(neighbors_dict_EF[index])
    x_EF_compressed = _ef_from_compressed_neighbors(neighbors_dict_EF_compressed[index], timesteps, n_static)
    x_latent = np.asarray(neighbors_dict_IF[index])

    s_static = _to_series(predict_for_samples(neighbors_dict_static[index]))
    s_ts = _to_series(predict_for_samples(neighbors_dict_ts[index]))
    s_EF = _to_series(predict_for_samples(x_EF))
    s_EF_compressed = _to_series(predict_for_samples(x_EF_compressed))
    s_frank = _to_series(predict_for_frankenstein(neighbors_dict_frankenstein[index]))
    s_latent = _to_series(predict_for_samples(x_latent))

    cols = [s_static, s_ts, s_EF, s_EF_compressed, s_frank, s_latent]
    col_names = ["static", "ts", "EF", "EF_compressed", "Frankenstein", "Latent"]

    if neighbors_dict_combined is not None:
        try:
            x_combined = _ef_from_combined_parts(neighbors_dict_combined[index])
            s_combined = _to_series(predict_for_samples(x_combined))
            cols.append(s_combined)
            col_names.append("Combined")
        except KeyError:
            pass

    neighbors_df = pd.concat(cols, axis=1)
    neighbors_df.columns = col_names
    return neighbors_df


# Progress bar helper (works well in notebooks)
from tqdm.auto import tqdm

# Evaluation utilities live in a separate module; import them here so
# `evaluate_methods_for_patient(s)` works when called from this module.
try:
    from counterfactual_evaluation_helpers import evaluate_counterfactuals, compute_tau_c, fit_plausibility_normalizer
except ImportError:  # support package-style imports
    from .counterfactual_evaluation_helpers import evaluate_counterfactuals, compute_tau_c, fit_plausibility_normalizer

if "traceback" not in globals():
    import traceback


def _ensure_eval_context(
    *,
    segments=None,
    tau_c=None,
    segment_length: int = 4,
    train_ts_obs=None,
):
    # Segments + tau_c (for ts sparsity objective)
    if segments is None:
        segments = globals().get("segments_modal", None)
    if segments is None:
        segments = globals().get("segments", None)
    if segments is None:
        if train_ts_obs is None:
            train_ts_obs = globals().get("train_ts", None)
        if train_ts_obs is None:
            raise ValueError("train_ts must be provided to derive default segments")
        if isinstance(train_ts_obs, dict):
            segments = {}
            for key in ("labs", "meds"):
                T = int(np.asarray(train_ts_obs[key]).shape[1])
                segments[key] = [np.arange(i, i + segment_length) for i in range(0, T, segment_length)]
        else:
            T = int(np.asarray(train_ts_obs).shape[1])
            segments = [np.arange(i, i + segment_length) for i in range(0, T, segment_length)]

    if tau_c is None:
        tau_c = globals().get("tau_c_modal", None)
    if tau_c is None:
        tau_c = globals().get("tau_c", None)
    if tau_c is None:
        if train_ts_obs is None:
            train_ts_obs = globals().get("train_ts", None)
        if train_ts_obs is None:
            raise ValueError("train_ts must be provided to derive default tau_c")
        tau_c = compute_tau_c(train_ts_obs)
        if isinstance(tau_c, dict):
            tau_c = {k: np.maximum(np.asarray(v, dtype=float), 1e-3) for k, v in tau_c.items()}
        else:
            tau_c = np.maximum(np.asarray(tau_c, dtype=float), 1e-3)

    return segments, tau_c


def evaluate_methods_for_patient(
    method_to_counterfactuals: dict,
    patient: int,
    *,
    train_static=None,
    train_ts=None,
    train_meds=None,
    train_labs=None,
    train_text=None,
    test_static=None,
    test_ts=None,
    test_meds=None,
    test_labs=None,
    test_text=None,
    y_test=None,
    trained_model=None,
    y_target=None,
    pred_to_label=None,
    rho: float = 0.0,
    segments=None,
    tau_c=None,
    segment_length: int = 4,
    text_objective_context=None,
    on_error: str = "collect",
    prediction_batch_size: int = 32,
    return_errors: bool = False,
) -> pd.DataFrame:
    """Evaluate multiple counterfactual-generation methods for one patient.

    Returns either:
      - df
      - (df, errors_df) if return_errors=True

    errors_df includes method name + full traceback.
    """

    if y_test is None:
        y_test = globals().get("y_test", None)
    if y_test is None:
        raise ValueError("y_test must be provided")

    if trained_model is None:
        trained_model = globals().get("trained_model", None)
    if trained_model is None:
        raise ValueError("trained_model must be provided")

    if train_static is None:
        train_static = globals().get("train_static", None)
    if train_ts is None:
        train_ts = globals().get("train_ts", None)
    if train_meds is None:
        train_meds = globals().get("train_meds", None)
    if train_labs is None:
        train_labs = globals().get("train_labs", None)
    if train_text is None:
        train_text = globals().get("train_text", None)
    if test_static is None:
        test_static = globals().get("test_static", None)
    if test_ts is None:
        test_ts = globals().get("test_ts", None)
    if test_meds is None:
        test_meds = globals().get("test_meds", None)
    if test_labs is None:
        test_labs = globals().get("test_labs", None)
    if test_text is None:
        test_text = globals().get("test_text", None)

    # Build split-ts dicts from explicit labs/meds inputs when provided.
    if train_ts is None and train_labs is not None and train_meds is not None:
        train_ts = {"labs": np.asarray(train_labs), "meds": np.asarray(train_meds)}
    if test_ts is None and test_labs is not None and test_meds is not None:
        test_ts = {"labs": np.asarray(test_labs), "meds": np.asarray(test_meds)}

    if train_static is None or train_ts is None or test_static is None or test_ts is None:
        raise ValueError(
            "train_static, train_ts, test_static, and test_ts must be provided "
            "(this function no longer relies on notebook globals)"
        )

    if y_target is None:
        y_target = 0 if int(y_test[patient]) == 1 else 1

    segments, tau_c = _ensure_eval_context(
        segments=segments,
        tau_c=tau_c,
        segment_length=segment_length,
        train_ts_obs=train_ts,
    )
    # Compute once per patient and reuse across methods to reduce repeated allocations/work.
    if text_objective_context is not None and "embedding_cache" not in text_objective_context:
        text_objective_context["embedding_cache"] = {}

    plausibility_normalizer = fit_plausibility_normalizer(train_static, train_ts)

    # Must match evaluate_counterfactuals()['objectives'] columns (N, 4)
    obj_cols = [
        "obj_outcome",
        "obj_multimodal_proximity",
        "obj_multimodal_sparsity",
        "obj_merged_plausibility",
    ]

    rows = []
    error_rows = []

    # Count total counterfactuals for progress tracking
    total_cfs = 0
    for method, cfs in method_to_counterfactuals.items():
        if isinstance(cfs, dict):
            total_cfs += len(cfs)
        else:
            total_cfs += len(cfs) if hasattr(cfs, '__len__') else 1
    evaluated_cfs = 0

    for method, cfs in method_to_counterfactuals.items():
        # Normalize dict->list while retaining IDs
        if isinstance(cfs, dict):
            cfs_in = list(cfs.values())
            cf_ids = list(cfs.keys())
        else:
            cfs_in = cfs
            cf_ids = None

        text_factual = None
        text_input_ref = None
        if test_text is not None:
            try:
                text_input_ref = np.asarray(test_text, dtype=object).reshape(-1)[int(patient)]
                text_factual = str(text_input_ref)
            except Exception:
                text_input_ref = None
                text_factual = None

        text_candidates = None
        text_input_candidates = None
        if isinstance(cfs_in, list) and len(cfs_in) > 0 and isinstance(cfs_in[0], dict):
            _txt = []
            _txt_in = []
            _has_txt = False
            _has_txt_in = False
            for item in cfs_in:
                t = item.get("text", None) if isinstance(item, dict) else None
                ti = item.get("text_input", None) if isinstance(item, dict) else None
                _txt.append(None if t is None else str(t))
                _txt_in.append(ti)
                _has_txt = _has_txt or (t is not None)
                _has_txt_in = _has_txt_in or (ti is not None)
            if _has_txt:
                text_candidates = _txt
            if _has_txt_in:
                text_input_candidates = _txt_in

        try:
            ev = evaluate_counterfactuals(
                x_tab_ref=test_static[patient],
                x_ts_ref=({k: np.asarray(v)[patient] for k, v in test_ts.items()} if isinstance(test_ts, dict) else test_ts[patient]),
                counterfactuals=cfs_in,
                X_tab_obs=train_static,
                X_ts_obs=train_ts,
                model=trained_model,
                segments=segments,
                tau_c=tau_c,
                y_target=y_target,
                rho=rho,
                plausibility_normalizer=plausibility_normalizer,
                pred_to_label=pred_to_label,
                text_factual=text_factual,
                text_candidates=text_candidates,
                text_objective_context=text_objective_context,
                text_input_ref=text_input_ref,
                text_input_candidates=text_input_candidates,
                prediction_batch_size=prediction_batch_size,
            )

            for i in range(len(ev["pred_cf"])):
                row = {
                    "method": method,
                    "patient": int(patient),
                    "cf_rank": int(i),
                    "cf_id": (str(cf_ids[i]) if cf_ids is not None else None),
                    "cf_kind": ev["cf_kind"][i],
                    "pred_factual": float(ev["pred_factual"]),
                    "label_factual": int(ev["label_factual"]),
                    "pred_cf": float(ev["pred_cf"][i]),
                    "label_cf": int(ev["label_cf"][i]),
                    "flipped": bool(ev["flipped"][i]),
                }
                for j, name in enumerate(obj_cols):
                    row[name] = float(ev["objectives"][i, j])
                rows.append(row)

            # Update progress
            method_cf_count = len(ev["pred_cf"])
            evaluated_cfs += method_cf_count
            progress_pct = (evaluated_cfs / total_cfs * 100) if total_cfs > 0 else 100
            if method_cf_count > 0 or evaluated_cfs == total_cfs:
                print(f"Patient {patient}: {progress_pct:.1f}% complete ({evaluated_cfs}/{total_cfs} counterfactuals)")

        except Exception as e:
            if on_error == "raise":
                raise
            error_rows.append(
                {
                    "patient": int(patient),
                    "method": str(method),
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

        # Update progress
        method_cf_count = len(ev["pred_cf"]) if 'ev' in locals() else 0
        evaluated_cfs += method_cf_count
        progress_pct = (evaluated_cfs / total_cfs * 100) if total_cfs > 0 else 100
        print(f"Patient {patient}: {progress_pct:.1f}% complete ({evaluated_cfs}/{total_cfs} counterfactuals)")

    print(f"Patient {patient}: evaluated {len(rows)} counterfactuals across {len(method_to_counterfactuals)} methods.")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values(["method", "flipped", "obj_outcome"], ascending=[True, False, True])

    errors_df = pd.DataFrame(error_rows)

    if return_errors:
        return df, errors_df
    return df


def evaluate_methods_for_patients(
    methods_to_cfs_by_patient: dict,
    y_test=None,
    trained_model=None,
    patients=None,
    *,
    train_static=None,
    train_ts=None,
    train_meds=None,
    train_labs=None,
    train_text=None,
    test_static=None,
    test_ts=None,
    test_meds=None,
    test_labs=None,
    test_text=None,
    pred_to_label=None,
    rho: float = 0.0,
    on_error: str = "collect",
    n_jobs: int = 1,
    parallel_backend: str = "threading",
    parallel_verbose: int = 0,
    text_objective_context=None,
    prediction_batch_size: int = 32,
):
    """Evaluate many patients when you have: {patient_idx: {method: counterfactuals}}.

    Returns
    -------
    (df_all, errors, errors_df)
      - df_all: concatenated per-CF evaluation rows
      - errors: dict keyed by (patient, method) -> short error string
      - errors_df: DataFrame with patient/method/error_type/error/traceback

    Notes
    -----
    This catches errors per method, so one failing method won't hide others.
    """

    if patients is None:
        patients = list(methods_to_cfs_by_patient.keys())
    else:
        patients = list(patients)

    print(f"Starting evaluation for {len(patients)} patients with {len(methods_to_cfs_by_patient[list(methods_to_cfs_by_patient.keys())[0]])} methods per patient.")

    dfs = []
    errors = {}
    error_dfs = []

    def _evaluate_one_patient(p):
        print(f"Evaluating patient {p}...")
        p_int = int(p)
        if p not in methods_to_cfs_by_patient and p_int in methods_to_cfs_by_patient:
            p_key = p_int
        else:
            p_key = p

        method_to_cfs = methods_to_cfs_by_patient.get(p_key, {})
        return evaluate_methods_for_patient(
            method_to_cfs,
            p_int,
            train_static=train_static,
            train_ts=train_ts,
            train_meds=train_meds,
            train_labs=train_labs,
            train_text=train_text,
            test_static=test_static,
            test_ts=test_ts,
            test_meds=test_meds,
            test_labs=test_labs,
            test_text=test_text,
            y_test = y_test,
            trained_model=trained_model,
            pred_to_label=pred_to_label,
            rho=rho,
            text_objective_context=text_objective_context,
            on_error=on_error,
            return_errors=True,
            prediction_batch_size=prediction_batch_size,
        )

    print(f"Starting evaluation of {len(patients)} patients...")

    if n_jobs is None:
        n_jobs = 1

    backend = str(parallel_backend)
    if trained_model is not None and backend in {"loky", "multiprocessing", "processes"}:
        # Keras models are not reliably pickle-safe across process workers.
        # Force threads to avoid model deserialization failures in joblib workers.
        backend = "threading"
        if int(parallel_verbose) > 0:
            print(
                "evaluate_methods_for_patients: switched backend to 'threading' "
                "because trained_model was provided."
            )

    if int(n_jobs) == 1:
        print("Running evaluations sequentially.")
        patient_results = [_evaluate_one_patient(p) for p in patients]
    else:
        print(f"Running evaluations in parallel with {n_jobs} jobs using '{backend}' backend.")
        from joblib import Parallel, delayed
        patient_results = Parallel(
            n_jobs=int(n_jobs),
            backend=backend,
            verbose=int(parallel_verbose),
        )(
            delayed(_evaluate_one_patient)(p) for p in patients
        )

    for df_p, errors_df_p in patient_results:
        if len(df_p) > 0:
            dfs.append(df_p)

        if len(errors_df_p) > 0:
            error_dfs.append(errors_df_p)
            for _, row in errors_df_p.iterrows():
                key = (int(row["patient"]), str(row["method"]))
                errors[key] = f"{row['error_type']}: {row['error']}"

    print(f"Evaluated {len(patients)} patients.")

    df_all = pd.concat(dfs, ignore_index=True) if len(dfs) > 0 else pd.DataFrame()
    errors_df = pd.concat(error_dfs, ignore_index=True) if len(error_dfs) > 0 else pd.DataFrame()

    print(f"Collected results: {len(df_all)} counterfactual evaluations, {len(errors)} errors.")

    return df_all, errors, errors_df
