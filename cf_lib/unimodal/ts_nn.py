"""Generic time-series nearest-neighbour counterfactual generator."""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
from aeon.distances import dtw_distance

from counterfactual_helpers import find_k_closest_ts
from cf_lib.base import CounterfactualGenerator


class TimeSeriesNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in a named time-series space.

    Searches for the k training samples closest to the query in the distance
    space of the specified time series (sum of per-channel distances).
    Progressive-median synthetic candidates are materialized and returned.

    Parameters
    ----------
    ts_name     : key in ``dataset.X_train_ts`` / ``dataset.X_test_ts``
    k           : default number of candidates
    distance_fn : optional per-channel distance callable (default: aeon dtw_distance)
    distance_metric : optional metric name {"dtw", "euclidean", "lcss"}.
                      When provided, it takes precedence over ``distance_fn``.
    dtw_window  : DTW warping window (default 0.10). Used when DTW is selected.
    distance_kwargs : optional extra kwargs forwarded to the underlying aeon
                      distance function.
    """

    def __init__(
        self,
        ts_name: str,
        k: int = 50,
        distance_fn: Callable = dtw_distance,
        distance_metric: Optional[str] = None,
        dtw_window: Optional[float] = 0.10,
        distance_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.ts_name = ts_name
        self.k = k
        self.distance_fn = distance_fn
        self.distance_metric = distance_metric
        self.dtw_window = dtw_window
        self.distance_kwargs = distance_kwargs

    # ------------------------------------------------------------------
    @staticmethod
    def _materialize(
        indices_dict: dict,
        sample_idx: int,
        dataset,
        ts_name: str,
        distance_metric_label: str,
    ) -> List[Dict[str, Any]]:
        """Build progressive-median candidates from neighbor indices.

        Each candidate contains the static median, the ts median for
        ``ts_name``, and (when present) text using rank-matched selection.
        """
        idxs = np.asarray(indices_dict.get(int(sample_idx), []), dtype=int)
        n_avail = idxs.size
        if n_avail == 0:
            return []

        train_static = np.asarray(dataset.X_train_static)
        train_ts = np.asarray(dataset.X_train_ts[ts_name])
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
            ts_med = np.asarray(np.median(train_ts[subset], axis=0), dtype=float)
            tab_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_tab.items()
            }
            text_val = str(train_text[anchor]) if train_text is not None else None
            text_input = train_text[anchor] if train_text is not None else None

            results.append(
                {
                    "static": static_med,
                    "ts": {ts_name: ts_med},
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
            metric = str(self.distance_metric)
            if str(self.distance_metric).strip().lower() == "dtw":
                return f"{metric}(window={self.dtw_window})"
            return metric
        if self.distance_fn is None:
            return f"dtw(window={self.dtw_window})"
        fn_name = getattr(self.distance_fn, "__name__", self.distance_fn.__class__.__name__)
        if fn_name == "dtw_distance":
            return f"dtw(window={self.dtw_window})"
        return fn_name

    # ------------------------------------------------------------------
    def generate_raw(
        self,
        dataset,
        sample_idx: int,
        target_value: int = 0,
        k: Optional[int] = None,
    ):
        """Run the NN search and return ``(candidates, indices_dict, distances_dict)``."""
        ts_train = (dataset.X_train_ts or {}).get(self.ts_name)
        ts_test = (dataset.X_test_ts or {}).get(self.ts_name)
        if ts_train is None or ts_test is None:
            print(
                f"[TimeSeriesNN:{self.ts_name!r}] ts_name not found in dataset "
                "— returning empty candidates."
            )
            return [], {}, {}

        k = k if k is not None else self.k
        indices, distances = find_k_closest_ts(
            X_train_ts=ts_train,
            y_train=dataset.y_train,
            X_test_ts=ts_test,
            selected_test_indices=[sample_idx],
            target_value=target_value,
            k=k,
            distance_fn=self.distance_fn,
            distance_metric=self.distance_metric,
            dtw_window=self.dtw_window,
            distance_kwargs=self.distance_kwargs,
            return_indices=True,
        )
        candidates = self._materialize(
            indices,
            sample_idx,
            dataset,
            self.ts_name,
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
        candidates, _, _ = self.generate_raw(
            dataset, sample_idx, target_value=target_value, k=k
        )
        return candidates
