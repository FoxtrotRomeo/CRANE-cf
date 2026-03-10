"""Medications time-series nearest-neighbour counterfactual generator.

Backward-compatible alias for ``TimeSeriesNN(ts_name="meds", ...)``.
Expects the dataset to have ``X_train_ts = {"meds": ..., ...}``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from aeon.distances import dtw_distance

from cf_lib.unimodal.ts_nn import TimeSeriesNN


class MedsNN(TimeSeriesNN):
    """Backward-compatible alias for ``TimeSeriesNN("meds")``."""

    def __init__(
        self,
        k: int = 50,
        distance_fn: Callable = dtw_distance,
        distance_metric: Optional[str] = None,
        dtw_window: Optional[float] = 0.10,
        distance_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            ts_name="meds",
            k=k,
            distance_fn=distance_fn,
            distance_metric=distance_metric,
            dtw_window=dtw_window,
            distance_kwargs=distance_kwargs,
        )
