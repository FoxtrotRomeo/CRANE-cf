"""Early-Fusion nearest-neighbour counterfactual generator."""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances

from cf_lib.base import CounterfactualGenerator


class EarlyFusionNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in a concatenated multimodal space.

    Builds a single joint representation by concatenating:
        - static features,
        - time-mean of **every** time-series modality in ``dataset.X_train_ts``
          (in insertion-key order, any number of TS supported),
        - E5 text embedding (when text is present and an embed function is provided).

    Distance is configurable through either a metric string
    (e.g., "euclidean", "manhattan", "hamming") or a callable.

    Parameters
    ----------
    k            : default number of candidates
    e5_tokenizer : HuggingFace tokenizer for the E5 model
    e5_model     : HuggingFace E5 model
    e5_device    : torch device string, e.g. "cuda" or "cpu"
    e5_embed_fn  : optional callable(texts) -> np.ndarray; overrides tokenizer/model.
                   When neither is provided the text modality is skipped even if present.
    distance_fn  : optional pairwise distance callable
    distance_metric : optional metric string; when provided it takes precedence
                      over ``distance_fn``.
    """

    def __init__(
        self,
        k: int = 20,
        e5_tokenizer=None,
        e5_model=None,
        e5_device=None,
        e5_embed_fn: Optional[Callable] = None,
        distance_fn: Optional[Callable] = None,
        distance_metric: Optional[str] = None,
    ):
        self.k = k
        self.e5_tokenizer = e5_tokenizer
        self.e5_model = e5_model
        self.e5_device = e5_device
        self.e5_embed_fn = e5_embed_fn
        self.distance_fn = distance_fn
        self.distance_metric = distance_metric

    # ------------------------------------------------------------------
    def _embed(self, texts) -> np.ndarray:
        """Return E5 embeddings for a list of raw texts."""
        strs = ["" if t is None else str(t) for t in np.asarray(texts, dtype=object).reshape(-1)]
        if self.e5_embed_fn is not None:
            return np.asarray(self.e5_embed_fn(strs), dtype=float)
        # Fall back to tokenizer/model if provided.
        try:
            from counterfactual_evaluation_helpers import embed_e5
        except ImportError:
            from counterfactual_evaluation_helpers import embed_e5
        return np.asarray(
            embed_e5(strs, tokenizer=self.e5_tokenizer, model=self.e5_model, device=self.e5_device),
            dtype=float,
        )

    def _has_text_support(self) -> bool:
        return (
            self.e5_embed_fn is not None
            or (self.e5_tokenizer is not None and self.e5_model is not None)
        )

    @staticmethod
    def _normalize_metric(metric: Optional[str], default: str = "euclidean") -> str:
        m = default if metric is None else str(metric).strip().lower()
        aliases = {
            "l2": "euclidean",
            "euclid": "euclidean",
            "l1": "manhattan",
            "cityblock": "manhattan",
            "manhattan_distance": "manhattan",
            "hamming_distance": "hamming",
        }
        return aliases.get(m, m)

    def _distance_matrix(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Return pairwise distances shaped (n_query, n_candidates)."""
        query = np.asarray(query, dtype=float)
        candidates = np.asarray(candidates, dtype=float)
        n_query = int(query.shape[0])
        n_cand = int(candidates.shape[0])

        if self.distance_metric is not None:
            metric = self._normalize_metric(self.distance_metric)
            return np.asarray(pairwise_distances(query, candidates, metric=metric), dtype=float)

        if self.distance_fn is None:
            return np.asarray(euclidean_distances(query, candidates), dtype=float)

        try:
            dm = np.asarray(self.distance_fn(query, candidates), dtype=float)
            if dm.shape == (n_query, n_cand):
                return dm
            if dm.shape == (n_cand, n_query):
                return dm.T
        except Exception:
            pass

        out = np.empty((n_query, n_cand), dtype=float)
        for r in range(n_query):
            s = query[r].reshape(1, -1)
            dists = None
            try:
                res = self.distance_fn(candidates, s)
                dists = np.asarray(res, dtype=float).ravel()
                if dists.size != n_cand:
                    try:
                        res2 = self.distance_fn(s, candidates)
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
                        val = self.distance_fn(candidates[i], s1)
                    except Exception:
                        val = self.distance_fn(s1, candidates[i])
                    arr = np.asarray(val, dtype=float).ravel()
                    dists[i] = float(arr[0] if arr.size > 0 else val)
            out[r] = dists
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _build_concat(static, ts_dict, tab_dict=None, emb=None) -> np.ndarray:
        """Concatenate [static, named-tabular…, time-mean(ts)…, emb].

        Named tabular arrays are already 2D ``(n, D)`` and are concatenated
        directly.  Time-series arrays are 3D ``(n, T, D)`` and reduced to 2D
        by taking the time-mean before concatenating.
        """
        parts = [np.asarray(static, dtype=float)]
        for arr in (tab_dict or {}).values():
            parts.append(np.asarray(arr, dtype=float))
        for arr in ts_dict.values():
            parts.append(np.mean(np.asarray(arr, dtype=float), axis=1))
        if emb is not None:
            parts.append(np.asarray(emb, dtype=float))
        return np.concatenate(parts, axis=1)

    # ------------------------------------------------------------------
    def generate(
        self,
        dataset,
        sample_idx: int,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        k = k if k is not None else self.k

        train_ts = dataset.X_train_ts or {}
        test_ts = dataset.X_test_ts or {}
        train_tab = dataset.X_train_tab or {}
        test_tab = dataset.X_test_tab or {}

        # --- text embeddings (optional) ---
        use_text = "text" in dataset.available_modalities and self._has_text_support()
        emb_train = self._embed(dataset.X_train_text) if use_text else None
        emb_test = self._embed(dataset.X_test_text) if use_text else None

        # --- build concatenated vectors ---
        X_train_concat = self._build_concat(dataset.X_train_static, train_ts, train_tab, emb_train)
        X_test_concat = self._build_concat(dataset.X_test_static, test_ts, test_tab, emb_test)

        # --- NN search filtered to target_value class ---
        y_train = np.asarray(dataset.y_train).reshape(-1)
        candidate_indices = np.flatnonzero(y_train == target_value)
        if candidate_indices.size == 0:
            return []

        query = X_test_concat[[sample_idx]]          # shape (1, D_concat)
        candidate_vecs = X_train_concat[candidate_indices]
        dists_row = self._distance_matrix(query, candidate_vecs)[0]

        k_eff = min(k, candidate_indices.size)
        if k_eff == candidate_indices.size:
            sorted_local = np.argsort(dists_row)[:k_eff]
        else:
            part = np.argpartition(dists_row, k_eff - 1)[:k_eff]
            sorted_local = part[np.argsort(dists_row[part])]

        neighbor_train_idx = candidate_indices[sorted_local]  # absolute train indices

        # --- materialize per-modality progressive-median candidates ---
        train_static = np.asarray(dataset.X_train_static, dtype=float)
        train_ts_np = {name: np.asarray(arr, dtype=float) for name, arr in train_ts.items()}
        train_tab_np = {name: np.asarray(arr, dtype=float) for name, arr in train_tab.items()}
        train_text = (
            np.asarray(dataset.X_train_text, dtype=object).reshape(-1)
            if dataset.X_train_text is not None
            else None
        )

        results = []
        for i in range(1, k_eff + 1):
            use_n = min(2 * i - 1, k_eff)
            subset = neighbor_train_idx[:use_n]
            anchor = int(neighbor_train_idx[i - 1])   # rank-matched: candidate i → neighbor i

            static_med = np.median(train_static[subset], axis=0)
            ts_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_ts_np.items()
            }
            tab_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_tab_np.items()
            }
            text_val = str(train_text[anchor]) if train_text is not None else None
            text_input = train_text[anchor] if train_text is not None else None

            results.append(
                {
                    "static": np.asarray(static_med, dtype=float),
                    "ts": ts_meds,
                    "tab": tab_meds,
                    "text": text_val,
                    "text_input": text_input,
                    "source_train_idx": anchor,
                    "n_neighbors_used": int(use_n),
                }
            )
        return results
