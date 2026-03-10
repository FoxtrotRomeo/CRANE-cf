"""Combined (joint-modality) nearest-neighbour counterfactual generator."""
from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
from aeon.distances import dtw_distance
from sklearn.metrics.pairwise import euclidean_distances

from counterfactual_helpers import find_k_closest_static, find_k_closest_ts, find_k_closest_text
from cf_lib.base import CounterfactualGenerator


class CombinedNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals with joint inter-modality agreement.

    Runs separate unimodal NN searches for static features, each time-series
    modality, and (optionally) text. Then intersects the candidate sets across
    all modalities, sums the per-modality distances for each common candidate,
    and selects the k with the lowest summed distance.

    Final candidates are materialised as progressive-median synthetic samples.

    Parameters
    ----------
    k               : default number of combined candidates
    k_search        : pool size used in each unimodal search before intersection
    static_dist_fn  : distance for the static search (default: euclidean_distances)
    ts_dist_fn      : per-channel distance for time-series (default: dtw_distance)
    e5_tokenizer    : HuggingFace tokenizer for the E5 model (text search)
    e5_model        : HuggingFace E5 model
    e5_device       : torch device string, e.g. "cuda" or "cpu"
    e5_embed_fn     : optional callable; overrides tokenizer/model
    text_repr_fn    : optional callable(raw_text) -> str for E5 input conversion
    """

    def __init__(
        self,
        k: int = 5,
        k_search: int = 50,
        static_dist_fn: Callable = euclidean_distances,
        ts_dist_fn: Callable = dtw_distance,
        e5_tokenizer=None,
        e5_model=None,
        e5_device=None,
        e5_embed_fn: Optional[Callable] = None,
        text_repr_fn: Optional[Callable] = None,
    ):
        self.k = k
        self.k_search = k_search
        self.static_dist_fn = static_dist_fn
        self.ts_dist_fn = ts_dist_fn
        self.e5_tokenizer = e5_tokenizer
        self.e5_model = e5_model
        self.e5_device = e5_device
        self.e5_embed_fn = e5_embed_fn
        self.text_repr_fn = text_repr_fn

    # ------------------------------------------------------------------
    @staticmethod
    def _combined_partial(active_idx_dist: dict, sample_idx: int, k: int, dataset) -> list:
        """Intersect + rank combination for an arbitrary subset of modalities.

        Parameters
        ----------
        active_idx_dist : dict role -> (idx_dict, dist_dict)
            ``"tabular"`` is always present; time-series modality names (any
            string keys from ``dataset.X_train_ts``) and ``"text"`` are optional.
        sample_idx : int
        k : int
        dataset : MultimodalDataset
        """
        if not active_idx_dist:
            return []

        maps: dict = {}
        for role, (idx_dict, dist_dict) in active_idx_dist.items():
            idxs = np.asarray(idx_dict.get(int(sample_idx), []), dtype=int).ravel()
            dists = np.asarray(dist_dict.get(int(sample_idx), []), dtype=float).ravel()
            maps[role] = {int(i): float(d) for i, d in zip(idxs, dists)}

        # Intersect all available index sets.
        common = set.intersection(*(set(m.keys()) for m in maps.values()))
        common_arr = np.array(sorted(common), dtype=int)
        if common_arr.size == 0:
            return []

        summed = np.array(
            [sum(m[int(i)] for m in maps.values()) for i in common_arr], dtype=float
        )
        k_eff = min(k, common_arr.size)
        if k_eff == common_arr.size:
            sel = np.argsort(summed)[:k_eff]
        else:
            kth = k_eff - 1
            part = np.argpartition(summed, kth)[:k_eff]
            sel = part[np.argsort(summed[part])]

        selected = common_arr[sel]

        # Gather train arrays.
        train_static = np.asarray(dataset.X_train_static)
        train_ts_dict = {
            ts_name: np.asarray(arr)
            for ts_name, arr in (dataset.X_train_ts or {}).items()
        }
        train_tab_dict = {
            tab_name: np.asarray(arr)
            for tab_name, arr in (dataset.X_train_tab or {}).items()
        }
        train_text = (
            np.asarray(dataset.X_train_text, dtype=object).reshape(-1)
            if dataset.X_train_text is not None
            else None
        )

        results = []
        n_avail = len(selected)
        for i in range(1, n_avail + 1):
            use_n = min(2 * i - 1, n_avail)
            subset = selected[:use_n]
            anchor = int(selected[i - 1])   # rank-matched: candidate i → neighbor i

            static_med = np.asarray(np.median(train_static[subset], axis=0), dtype=float)
            ts_meds = {
                ts_name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for ts_name, arr in train_ts_dict.items()
            }
            tab_meds = {
                tab_name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for tab_name, arr in train_tab_dict.items()
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
                }
            )
        return results

    # ------------------------------------------------------------------
    def _to_str_list(self, text_array) -> list:
        arr = np.asarray(text_array, dtype=object).reshape(-1)
        if self.text_repr_fn is not None:
            return [self.text_repr_fn(t) for t in arr]
        return ["" if t is None else str(t) for t in arr]

    # ------------------------------------------------------------------
    def generate(
        self,
        dataset,
        sample_idx: int,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
        precomputed: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate Combined-NN counterfactuals.

        Parameters
        ----------
        precomputed : optional dict with any of the following keys to skip the
            corresponding internal unimodal search::

                {
                    "tabular":   (indices_dict, distances_dict),
                    <ts_name>:   (indices_dict, distances_dict),  # one per ts key
                    "text":      (indices_dict, distances_dict),
                    "train_text_str": [...],
                }

            Missing keys are computed on-the-fly.
        """
        k = k if k is not None else self.k
        pc = precomputed or {}
        avail = dataset.available_modalities

        # --- static NN (always required) ---
        if "tabular" in pc:
            idx_static, dist_static = pc["tabular"]
        else:
            idx_static, dist_static = find_k_closest_static(
                X_train_static=dataset.X_train_static,
                y_train=dataset.y_train,
                X_test_static=dataset.X_test_static,
                selected_test_indices=[sample_idx],
                target_value=target_value,
                k=self.k_search,
                distance_fn=self.static_dist_fn,
                return_indices=True,
            )

        # --- per-ts NN (one search per key in X_train_ts) ---
        ts_idx_dist: Dict[str, tuple] = {}
        for ts_name in list((dataset.X_train_ts or {}).keys()):
            if ts_name in avail:
                if ts_name in pc:
                    ts_idx_dist[ts_name] = pc[ts_name]
                else:
                    idx_ts, dist_ts = find_k_closest_ts(
                        X_train_ts=dataset.X_train_ts[ts_name],
                        y_train=dataset.y_train,
                        X_test_ts=dataset.X_test_ts[ts_name],
                        selected_test_indices=[sample_idx],
                        target_value=target_value,
                        k=self.k_search,
                        distance_fn=self.ts_dist_fn,
                        return_indices=True,
                    )
                    ts_idx_dist[ts_name] = (idx_ts, dist_ts)

        # --- per-named-tabular NN (one search per key in X_train_tab) ---
        tab_idx_dist: Dict[str, tuple] = {}
        for tab_name in list((dataset.X_train_tab or {}).keys()):
            if tab_name in avail:
                if tab_name in pc:
                    tab_idx_dist[tab_name] = pc[tab_name]
                else:
                    idx_tab, dist_tab = find_k_closest_static(
                        X_train_static=dataset.X_train_tab[tab_name],
                        y_train=dataset.y_train,
                        X_test_static=dataset.X_test_tab[tab_name],
                        selected_test_indices=[sample_idx],
                        target_value=target_value,
                        k=self.k_search,
                        distance_fn=self.static_dist_fn,
                        return_indices=True,
                    )
                    tab_idx_dist[tab_name] = (idx_tab, dist_tab)

        # --- text NN (skipped when not in dataset) ---
        idx_text = dist_text = None
        if "text" in avail:
            train_text_str = pc.get("train_text_str") or self._to_str_list(dataset.X_train_text)
            test_text_str = self._to_str_list(dataset.X_test_text)
            if "text" in pc:
                idx_text, dist_text = pc["text"]
            else:
                ts_list = list((dataset.X_train_ts or {}).values())
                train_ts1 = ts_list[0] if len(ts_list) > 0 else None
                train_ts2 = ts_list[1] if len(ts_list) > 1 else None
                ts_test_list = list((dataset.X_test_ts or {}).values())
                test_ts1 = ts_test_list[0] if len(ts_test_list) > 0 else None
                test_ts2 = ts_test_list[1] if len(ts_test_list) > 1 else None
                _, idx_text, dist_text = find_k_closest_text(
                    X_train_static=dataset.X_train_static,
                    X_train_meds=train_ts1,
                    X_train_labs=train_ts2,
                    train_text_for_distance=train_text_str,
                    y_train=dataset.y_train,
                    X_test_static=dataset.X_test_static,
                    X_test_meds=test_ts1,
                    X_test_labs=test_ts2,
                    test_text_for_distance=test_text_str,
                    selected_test_indices=[sample_idx],
                    target_value=target_value,
                    k=self.k_search,
                    e5_embed_fn=self.e5_embed_fn,
                    e5_tokenizer=self.e5_tokenizer,
                    e5_model=self.e5_model,
                    e5_device=self.e5_device,
                    train_text_raw=dataset.X_train_text,
                    test_text_raw=dataset.X_test_text,
                )

        # --- combine: intersect + rank ---
        active_idx_dist: dict = {"tabular": (idx_static, dist_static)}
        for ts_name, (idx_ts, dist_ts) in ts_idx_dist.items():
            active_idx_dist[ts_name] = (idx_ts, dist_ts)
        for tab_name, (idx_tab, dist_tab) in tab_idx_dist.items():
            active_idx_dist[tab_name] = (idx_tab, dist_tab)
        if idx_text is not None:
            active_idx_dist["text"] = (idx_text, dist_text)

        return self._combined_partial(active_idx_dist, sample_idx, k, dataset)
