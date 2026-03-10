"""Intermediate-Fusion nearest-neighbour counterfactual generator."""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from sklearn.metrics.pairwise import euclidean_distances

from counterfactual_helpers import find_k_closest_latent_model
from cf_lib.base import CounterfactualGenerator


class IntermediateFusionNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in the classifier's latent space.

    Builds a latent model from the penultimate layer of the provided Keras
    classifier, computes embeddings for all train/test samples, then selects
    the k closest opposite-class training samples in that latent space.

    Parameters
    ----------
    k            : default number of candidates
    ts1_name     : key in ``dataset.X_train_ts`` to use as the first ts input
                   to the model (defaults to the first key in insertion order)
    ts2_name     : key in ``dataset.X_train_ts`` to use as the second ts input
                   (defaults to the second key in insertion order, or None)
    distance_fn  : latent-space distance callable (default: sklearn euclidean_distances)
    distance_metric : optional metric string (e.g., "euclidean", "manhattan", "hamming");
                      when provided it takes precedence over ``distance_fn``.
    """

    def __init__(
        self,
        k: int = 5,
        ts1_name: Optional[str] = None,
        ts2_name: Optional[str] = None,
        distance_fn: Callable = euclidean_distances,
        distance_metric: Optional[str] = None,
    ):
        self.k = k
        self.ts1_name = ts1_name
        self.ts2_name = ts2_name
        self.distance_fn = distance_fn
        self.distance_metric = distance_metric

    def _resolve_ts(self, dataset, split: str = "train"):
        """Return (arr1, arr2) for the configured ts names, or None when absent."""
        ts = (dataset.X_train_ts if split == "train" else dataset.X_test_ts) or {}
        keys = list(ts.keys())
        name1 = self.ts1_name or (keys[0] if len(keys) > 0 else None)
        name2 = self.ts2_name or (keys[1] if len(keys) > 1 else None)
        return ts.get(name1), ts.get(name2)

    def generate(
        self,
        dataset,
        sample_idx: int,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if model is None:
            raise ValueError("IntermediateFusionNN requires a trained model.")
        k = k if k is not None else self.k

        train_ts1, train_ts2 = self._resolve_ts(dataset, "train")
        test_ts1, test_ts2 = self._resolve_ts(dataset, "test")

        neighbors, _, _ = find_k_closest_latent_model(
            X_train=dataset.X_train_static,
            y_train=dataset.y_train,
            X_test=dataset.X_test_static,
            selected_test_indices=[sample_idx],
            model=model,
            X_train_meds=train_ts1,
            X_train_labs=train_ts2,
            X_train_text=dataset.X_train_text,
            X_test_meds=test_ts1,
            X_test_labs=test_ts2,
            X_test_text=dataset.X_test_text,
            target_value=target_value,
            k=k,
            distance_fn=self.distance_fn,
            distance_metric=self.distance_metric,
            return_latents=True,
        )
        return neighbors.get(int(sample_idx), [])
