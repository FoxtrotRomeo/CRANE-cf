"""Text nearest-neighbour counterfactual generator."""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np

from counterfactual_helpers import find_k_closest_text
from cf_lib.base import CounterfactualGenerator


class TextNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in configurable text spaces.

    Supports embedding-based and direct text matching:
    - Embedding encoders: E5, BERT, TF-IDF, Word2Vec, or custom embedding fn.
    - Vector-space metrics: cosine, euclidean, manhattan, hamming, etc.
    - Direct text metrics: BLEU, ROUGE-1/2/L, LCS.

    Numeric modalities (static + all time-series) are built as odd-k
    progressive medians. Text uses rank-matched selection: candidate i
    draws text from the i-th nearest neighbor.

    Parameters
    ----------
    k                   : default number of candidates
    text_encoder        : "e5" (default), "bert", "tfidf", "word2vec", "custom", "raw"
    text_distance_metric: vector metric ("cosine", "euclidean", ...) or direct text
                          metric ("bleu", "rouge1", "rouge2", "rouge_l", "lcs")
    text_distance_fn    : optional custom distance callable
    text_embed_fn       : optional custom embed callable texts -> 2D np.ndarray
    e5_tokenizer/model/device/e5_embed_fn
                        : E5 backend configuration
    bert_tokenizer/model/device/bert_embed_fn
                        : BERT backend configuration
    tfidf_vectorizer/tfidf_kwargs
                        : TF-IDF backend configuration
    word2vec_model/word2vec_embed_fn
                        : Word2Vec backend configuration
    text_metric_kwargs  : optional kwargs for direct text metrics
    text_repr_fn        : optional callable(raw_text) -> str that converts each entry in
                          X_train/test_text to plain string. When ``None`` the
                      raw values are cast with ``str()``.
    """

    def __init__(
        self,
        k: int = 20,
        text_encoder: str = "e5",
        text_distance_metric: str = "cosine",
        text_distance_fn: Optional[Callable] = None,
        text_embed_fn: Optional[Callable] = None,
        e5_tokenizer=None,
        e5_model=None,
        e5_device=None,
        e5_embed_fn: Optional[Callable] = None,
        bert_tokenizer=None,
        bert_model=None,
        bert_device=None,
        bert_embed_fn: Optional[Callable] = None,
        bert_batch_size: int = 32,
        bert_max_length: int = 512,
        tfidf_vectorizer=None,
        tfidf_kwargs: Optional[Dict[str, Any]] = None,
        word2vec_model=None,
        word2vec_embed_fn: Optional[Callable] = None,
        text_metric_kwargs: Optional[Dict[str, Any]] = None,
        text_repr_fn: Optional[Callable] = None,
    ):
        self.k = k
        self.text_encoder = text_encoder
        self.text_distance_metric = text_distance_metric
        self.text_distance_fn = text_distance_fn
        self.text_embed_fn = text_embed_fn
        self.e5_tokenizer = e5_tokenizer
        self.e5_model = e5_model
        self.e5_device = e5_device
        self.e5_embed_fn = e5_embed_fn
        self.bert_tokenizer = bert_tokenizer
        self.bert_model = bert_model
        self.bert_device = bert_device
        self.bert_embed_fn = bert_embed_fn
        self.bert_batch_size = bert_batch_size
        self.bert_max_length = bert_max_length
        self.tfidf_vectorizer = tfidf_vectorizer
        self.tfidf_kwargs = tfidf_kwargs
        self.word2vec_model = word2vec_model
        self.word2vec_embed_fn = word2vec_embed_fn
        self.text_metric_kwargs = text_metric_kwargs
        self.text_repr_fn = text_repr_fn

    # ------------------------------------------------------------------
    def _to_str_list(self, text_array) -> list:
        arr = np.asarray(text_array, dtype=object).reshape(-1)
        if self.text_repr_fn is not None:
            return [self.text_repr_fn(t) for t in arr]
        return ["" if t is None else str(t) for t in arr]

    # ------------------------------------------------------------------
    @staticmethod
    def _materialize(
        indices_dict: dict,
        sample_idx: int,
        dataset,
        train_text_raw,
        distance_metric_label: str,
        text_encoder_label: str,
    ) -> List[Dict[str, Any]]:
        """Build progressive-median candidate dicts from neighbor indices.

        Text uses rank-matched selection: candidate i draws text from the
        i-th nearest neighbor.
        """
        idxs = np.asarray(indices_dict.get(int(sample_idx), []), dtype=int)
        n_avail = idxs.size
        if n_avail == 0:
            return []

        train_static = np.asarray(dataset.X_train_static, dtype=float)
        train_ts = {
            name: np.asarray(arr, dtype=float)
            for name, arr in (dataset.X_train_ts or {}).items()
        }
        train_tab = {
            name: np.asarray(arr, dtype=float)
            for name, arr in (dataset.X_train_tab or {}).items()
        }
        train_text_np = np.asarray(train_text_raw, dtype=object).reshape(-1)

        results = []
        for i in range(1, n_avail + 1):
            use_n = min(2 * i - 1, n_avail)
            subset = idxs[:use_n]
            anchor = int(idxs[i - 1])   # rank-matched: candidate i → neighbor i

            static_med = np.asarray(np.median(train_static[subset], axis=0), dtype=float)
            ts_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_ts.items()
            }
            tab_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_tab.items()
            }

            results.append(
                {
                    "static": static_med,
                    "ts": ts_meds,
                    "tab": tab_meds,
                    "text": str(train_text_np[anchor]),
                    "text_input": train_text_np[anchor],
                    "source_train_idx": anchor,
                    "n_neighbors_used": int(use_n),
                    "distance_metric": distance_metric_label,
                    "text_encoder": text_encoder_label,
                }
            )
        return results

    def _resolve_distance_metric_label(self) -> str:
        if self.text_distance_metric is not None:
            return str(self.text_distance_metric)
        if self.text_distance_fn is None:
            return "cosine"
        return getattr(self.text_distance_fn, "__name__", self.text_distance_fn.__class__.__name__)

    def _resolve_text_encoder_label(self) -> str:
        if self.text_encoder is not None:
            return str(self.text_encoder)
        if self.text_embed_fn is not None:
            return "custom"
        return "e5"

    # ------------------------------------------------------------------
    def generate_raw(
        self,
        dataset,
        sample_idx: int,
        target_value: int = 0,
        k: Optional[int] = None,
    ):
        """Run the text NN search.

        Returns ``(candidates, indices_dict, distances_dict, train_text_str)``.
        """
        if dataset.X_train_text is None or dataset.X_test_text is None:
            print("[TextNN] text modality not present in dataset — returning empty candidates.")
            return [], {}, {}, []

        k = k if k is not None else self.k
        train_text_str = self._to_str_list(dataset.X_train_text)
        test_text_str = self._to_str_list(dataset.X_test_text)

        # find_k_closest_text expects meds/labs arrays for candidate construction;
        # TextNN materializes candidates itself, so placeholders are fine when absent.
        n_train = int(np.asarray(dataset.X_train_static).shape[0])
        n_test = int(np.asarray(dataset.X_test_static).shape[0])
        ts_train_values = list((dataset.X_train_ts or {}).values())
        ts_test_values = list((dataset.X_test_ts or {}).values())
        train_meds = ts_train_values[0] if len(ts_train_values) > 0 else np.zeros((n_train, 1, 1), dtype=float)
        train_labs = ts_train_values[1] if len(ts_train_values) > 1 else np.zeros((n_train, 1, 1), dtype=float)
        test_meds = ts_test_values[0] if len(ts_test_values) > 0 else np.zeros((n_test, 1, 1), dtype=float)
        test_labs = ts_test_values[1] if len(ts_test_values) > 1 else np.zeros((n_test, 1, 1), dtype=float)

        _, indices_dict, distances_dict = find_k_closest_text(
            X_train_static=dataset.X_train_static,
            X_train_meds=train_meds,
            X_train_labs=train_labs,
            train_text_for_distance=train_text_str,
            y_train=dataset.y_train,
            X_test_static=dataset.X_test_static,
            X_test_meds=test_meds,
            X_test_labs=test_labs,
            test_text_for_distance=test_text_str,
            selected_test_indices=[sample_idx],
            target_value=target_value,
            k=k,
            text_encoder=self.text_encoder,
            text_distance_metric=self.text_distance_metric,
            text_distance_fn=self.text_distance_fn,
            text_metric_kwargs=self.text_metric_kwargs,
            text_embed_fn=self.text_embed_fn,
            e5_embed_fn=self.e5_embed_fn,
            e5_tokenizer=self.e5_tokenizer,
            e5_model=self.e5_model,
            e5_device=self.e5_device,
            bert_embed_fn=self.bert_embed_fn,
            bert_tokenizer=self.bert_tokenizer,
            bert_model=self.bert_model,
            bert_device=self.bert_device,
            bert_batch_size=self.bert_batch_size,
            bert_max_length=self.bert_max_length,
            tfidf_vectorizer=self.tfidf_vectorizer,
            tfidf_kwargs=self.tfidf_kwargs,
            word2vec_embed_fn=self.word2vec_embed_fn,
            word2vec_model=self.word2vec_model,
            train_text_raw=dataset.X_train_text,
            test_text_raw=dataset.X_test_text,
        )

        candidates = self._materialize(
            indices_dict,
            sample_idx,
            dataset,
            dataset.X_train_text,
            distance_metric_label=self._resolve_distance_metric_label(),
            text_encoder_label=self._resolve_text_encoder_label(),
        )
        return candidates, indices_dict, distances_dict, train_text_str

    def generate(
        self,
        dataset,
        sample_idx: int,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        candidates, _, _, _ = self.generate_raw(
            dataset, sample_idx, target_value=target_value, k=k
        )
        return candidates
