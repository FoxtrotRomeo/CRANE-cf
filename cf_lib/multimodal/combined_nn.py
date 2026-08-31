"""Combined (joint-modality) nearest-neighbour counterfactual generator."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from aeon.distances import dtw_distance
from sklearn.metrics.pairwise import euclidean_distances

from cf_lib.counterfactual_helpers import (
    find_k_closest_static,
    find_k_closest_ts,
    find_k_closest_text,
    find_k_closest_image,
)
from cf_lib.base import CounterfactualGenerator


class CombinedNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals with joint inter-modality agreement.

    Runs separate unimodal NN searches for each present modality (static
    tabular, each time-series key, named-tabular blocks, text, image).
    Intersects the candidate sets across all modalities, sums the
    per-modality distances for each common candidate, and selects the *k*
    with the lowest summed distance.

    All modalities are optional; only those present in the dataset are searched.
    Final candidates are materialised as progressive-median synthetic samples
    for numeric modalities; text and image use rank-matched selection.

    Parameters
    ----------
    k               : default number of combined candidates
    k_search        : pool size used in each unimodal search before intersection
    static_dist_fn  : distance for the static search (default: euclidean_distances)
    ts_dist_fn      : per-channel distance for time-series (default: dtw_distance)
    img_embed_fn    : optional callable for image encoding
    image_encoder   : backbone string for image encoding (default: ``"precomputed"``)
    img_distance_metric : distance metric for image search (default: ``"cosine"``)
    img_device      : torch device string for image encoding
    img_batch_size  : images per forward pass
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
        img_embed_fn: Optional[Callable] = None,
        image_encoder: str = "precomputed",
        img_distance_metric: str = "cosine",
        img_device: Optional[str] = None,
        img_batch_size: int = 32,
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
        self.img_embed_fn = img_embed_fn
        self.image_encoder = image_encoder
        self.img_distance_metric = img_distance_metric
        self.img_device = img_device
        self.img_batch_size = img_batch_size
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
            Any combination of ``"tabular"``, time-series / named-tabular keys,
            ``"text"``, and ``"image"``.
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
        train_static = (
            np.asarray(dataset.X_train_static)
            if dataset.X_train_static is not None else None
        )
        train_ts_dict = {
            ts_name: np.asarray(arr)
            for ts_name, arr in (dataset.X_train_ts or {}).items()
        }
        train_tab_dict = {
            tab_name: np.asarray(arr)
            for tab_name, arr in (dataset.X_train_tab or {}).items()
        }
        train_texts = {
            name: np.asarray(arr, dtype=object).reshape(-1)
            for name, arr in dataset.text_modalities("train").items()
        }
        train_images = dataset.image_modalities("train")

        results = []
        n_avail = len(selected)
        for i in range(1, n_avail + 1):
            use_n = min(2 * i - 1, n_avail)
            subset = selected[:use_n]
            anchor = int(selected[i - 1])   # rank-matched: candidate i → neighbor i

            static_med = (
                np.asarray(np.median(train_static[subset], axis=0), dtype=float)
                if train_static is not None else None
            )
            ts_meds = {
                ts_name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for ts_name, arr in train_ts_dict.items()
            }
            tab_meds = {
                tab_name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for tab_name, arr in train_tab_dict.items()
            }
            text_vals = {name: str(arr[anchor]) for name, arr in train_texts.items()}
            text_inputs = {name: arr[anchor] for name, arr in train_texts.items()}
            image_vals = {name: arr[anchor] for name, arr in train_images.items()}
            image_inputs = dict(image_vals)
            text_val = text_vals.get(dataset.primary_text_name)
            text_input = text_inputs.get(dataset.primary_text_name)
            img_val = image_vals.get(dataset.primary_image_name)

            results.append(
                {
                    "static": static_med,
                    "ts": ts_meds,
                    "tab": tab_meds,
                    "text": text_val,
                    "text_input": text_input,
                    "texts": text_vals,
                    "text_inputs": text_inputs,
                    "image": img_val,
                    "image_input": img_val,
                    "images": image_vals,
                    "image_inputs": image_inputs,
                    "source_train_idx": anchor,
                    "source_indices": {
                        "tabular": (
                            {name: anchor for name in train_tab_dict}
                            | ({dataset.primary_tabular_name: anchor} if train_static is not None else {})
                        ),
                        "ts": {name: anchor for name in train_ts_dict},
                        "text": {name: anchor for name in train_texts},
                        "image": {name: anchor for name in train_images},
                    },
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
                    "tabular":        (indices_dict, distances_dict),
                    <ts_name>:        (indices_dict, distances_dict),
                    ("text", name):   (indices_dict, distances_dict),
                    "train_text_str": {name: [...]},
                    ("image", name):  (indices_dict, distances_dict),
                }

            Missing keys are computed on-the-fly.
        """
        k = k if k is not None else self.k
        pc = precomputed or {}
        avail = dataset.available_modalities

        # --- static NN (only when tabular modality is present) ---
        active_idx_dist: dict = {}
        if "tabular" in avail:
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
            active_idx_dist["tabular"] = (idx_static, dist_static)

        # --- per-ts NN ---
        for ts_name in list((dataset.X_train_ts or {}).keys()):
            if ts_name in avail:
                if ts_name in pc:
                    active_idx_dist[ts_name] = pc[ts_name]
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
                    active_idx_dist[ts_name] = (idx_ts, dist_ts)

        # --- per-named-tabular NN ---
        for tab_name in list((dataset.X_train_tab or {}).keys()):
            if tab_name in avail:
                if tab_name in pc:
                    active_idx_dist[tab_name] = pc[tab_name]
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
                    active_idx_dist[tab_name] = (idx_tab, dist_tab)

        # --- text NN ---
        train_text_cache = pc.get("train_text_str", {})
        for text_name in dataset.text_names():
            cache_key = ("text", text_name)
            train_text_raw = dataset.get_text_branch(text_name, split="train")
            test_text_raw = dataset.get_text_branch(text_name, split="test")
            if train_text_raw is None or test_text_raw is None:
                continue
            train_text_str = train_text_cache.get(text_name) or self._to_str_list(train_text_raw)
            test_text_str = self._to_str_list(test_text_raw)
            if cache_key in pc:
                active_idx_dist[cache_key] = pc[cache_key]
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
                    train_text_raw=train_text_raw,
                    test_text_raw=test_text_raw,
                )
                active_idx_dist[cache_key] = (idx_text, dist_text)

        # --- image NN ---
        for image_name in dataset.image_names():
            cache_key = ("image", image_name)
            train_img = dataset.get_image_branch(image_name, split="train")
            test_img = dataset.get_image_branch(image_name, split="test")
            if train_img is None or test_img is None:
                continue
            if cache_key in pc:
                active_idx_dist[cache_key] = pc[cache_key]
            else:
                idx_image, dist_image = find_k_closest_image(
                    X_train_img=train_img,
                    y_train=dataset.y_train,
                    X_test_img=test_img,
                    selected_test_indices=[sample_idx],
                    target_value=target_value,
                    k=self.k_search,
                    image_encoder=self.image_encoder,
                    image_distance_metric=self.img_distance_metric,
                    embed_fn=self.img_embed_fn,
                    device=self.img_device,
                    batch_size=self.img_batch_size,
                )
                active_idx_dist[cache_key] = (idx_image, dist_image)

        if not active_idx_dist:
            print("[CombinedNN] no modalities available for this dataset — returning empty candidates.")
            return []

        return self._combined_partial(active_idx_dist, sample_idx, k, dataset)
