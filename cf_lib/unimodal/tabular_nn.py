"""Tabular (static) nearest-neighbour counterfactual generator."""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from counterfactual_helpers import find_k_closest_static
from cf_lib.base import CounterfactualGenerator


class TabularNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in a tabular feature space.

    When ``tab_name`` is ``None`` (default) the search is performed on the
    primary ``dataset.X_train_static`` array, identical to the original
    behaviour.  When ``tab_name`` is a string, the search is performed on
    ``dataset.X_train_tab[tab_name]`` instead — this is the mechanism for
    supporting an arbitrary number of named tabular modalities.

    Finds the k training samples closest to the query sample in the configured
    distance metric space, filtered to ``target_value`` class. Progressive-median
    synthetic candidates are returned.

    Text selection uses rank-matched strategy: candidate i draws text from
    the i-th nearest neighbor.

    Parameters
    ----------
    tab_name     : key in ``dataset.X_train_tab`` / ``dataset.X_test_tab``,
                   or ``None`` to use the primary ``X_train_static``.
    k            : default number of candidates
    distance_fn  : optional pairwise distance callable
    distance_metric : optional metric string (e.g., "euclidean", "manhattan", "hamming");
                      when provided it takes precedence over ``distance_fn``.
    """

    def __init__(
        self,
        tab_name: Optional[str] = None,
        k: int = 50,
        distance_fn: Optional[Callable] = euclidean_distances,
        distance_metric: Optional[str] = None,
    ):
        self.tab_name = tab_name
        self.k = k
        self.distance_fn = distance_fn
        self.distance_metric = distance_metric

    # ------------------------------------------------------------------
    @staticmethod
    def _materialize(
        indices_dict: dict,
        sample_idx: int,
        dataset,
        distance_metric_label: str,
    ) -> List[Dict[str, Any]]:
        """Build progressive-median candidate dicts from neighbor indices.

        All time-series modalities in ``dataset.X_train_ts`` and all named
        tabular modalities in ``dataset.X_train_tab`` are included.
        Text (when present) uses rank-matched selection: candidate i takes
        text from the i-th nearest neighbor.
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
        train_text = (
            np.asarray(dataset.X_train_text, dtype=object).reshape(-1)
            if dataset.X_train_text is not None
            else None
        )

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
            text_val = str(train_text[anchor]) if train_text is not None else None
            text_input = train_text[anchor] if train_text is not None else None

            results.append(
                {
                    "static": static_med,
                    "ts": ts_meds,
                    "tab": tab_meds,
                    "text": text_val,
                    "text_input": text_input,
                    "source_train_idx": anchor,
                    "n_neighbors_used": int(use_n),
                    "distance_metric": distance_metric_label,
                }
            )
        return results

    def _resolve_distance_metric_label(self) -> str:
        if self.distance_metric is not None:
            return str(self.distance_metric)
        if self.distance_fn is None:
            return "euclidean"
        return getattr(self.distance_fn, "__name__", self.distance_fn.__class__.__name__)

    # ------------------------------------------------------------------
    def generate_raw(
        self,
        dataset,
        sample_idx: int,
        target_value: int = 0,
        k: Optional[int] = None,
    ):
        """Run the NN search and return ``(candidates, indices_dict, distances_dict)``."""
        k = k if k is not None else self.k

        if self.tab_name is None:
            # Primary tabular: search on X_train_static (original behaviour).
            X_train_arr = dataset.X_train_static
            X_test_arr = dataset.X_test_static
        else:
            # Named tabular: search on X_train_tab[tab_name].
            X_train_tab = dataset.X_train_tab or {}
            X_test_tab = dataset.X_test_tab or {}
            if self.tab_name not in X_train_tab or self.tab_name not in X_test_tab:
                print(
                    f"[TabularNN:{self.tab_name!r}] tab_name not found in dataset "
                    "— returning empty candidates."
                )
                return [], {}, {}
            X_train_arr = X_train_tab[self.tab_name]
            X_test_arr = X_test_tab[self.tab_name]

        indices, distances = find_k_closest_static(
            X_train_static=X_train_arr,
            y_train=dataset.y_train,
            X_test_static=X_test_arr,
            selected_test_indices=[sample_idx],
            target_value=target_value,
            k=k,
            distance_fn=self.distance_fn,
            distance_metric=self.distance_metric,
            return_indices=True,
        )
        candidates = self._materialize(
            indices,
            sample_idx,
            dataset,
            distance_metric_label=self._resolve_distance_metric_label(),
        )
        return candidates, indices, distances

    def generate(
        self,
        dataset,
        sample_idx: int,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        candidates, _, _ = self.generate_raw(dataset, sample_idx, target_value=target_value, k=k)
        return candidates
