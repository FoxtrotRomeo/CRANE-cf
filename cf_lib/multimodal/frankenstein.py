"""Modality-wise Prototype Synthesis (MPS): independent per-modality nearest-neighbour counterfactual generator."""
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


class ModalityWisePrototypeSynthesis(CounterfactualGenerator):
    """Hybrid counterfactuals from independent per-modality neighbor searches.

    For each test sample, runs separate NN searches for each present modality
    (static tabular, each time-series key, named-tabular blocks, text, image).
    Candidates are assembled by taking the progressive median of each numeric
    modality's own neighbors independently — the *i*-th candidate is built
    from the *i* nearest neighbors per numeric modality, without requiring the
    same training sample to be close in all spaces.  Text and image use
    rank-matched selection (candidate *i* draws from the *i*-th neighbor).

    All modalities are optional; only the modalities present in the dataset
    are searched.

    Parameters
    ----------
    k               : default number of Frankenstein candidates to build
    k_search        : pool size used in each per-modality unimodal search
    static_dist_fn  : distance for the static search (default: euclidean_distances)
    ts_dist_fn      : per-channel distance for time-series (default: dtw_distance)
    img_embed_fn    : optional callable for image encoding; passed to
                      ``find_k_closest_image`` as ``embed_fn``
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
    def _frankenstein_partial(
        active_numeric: dict,
        active_texts: dict,
        active_images: dict,
        k: int,
        ts_names: Optional[set] = None,
        tab_names: Optional[set] = None,
        primary_text_name: str = "__primary__",
        primary_image_name: str = "__primary__",
    ) -> list:
        """Frankenstein combination for an arbitrary subset of modalities.

        Parameters
        ----------
        active_numeric : dict role -> (idx_array_for_sample, train_array)
            Any combination of ``"static"``, time-series modality names, and
            named tabular modality names.
            ``idx_array_for_sample`` is a 1-D int array of sorted neighbor indices.
        active_texts : dict[text_name -> (idx_array_for_sample, train_text_raw)]
        active_images : dict[image_name -> (idx_array_for_sample, train_img)]
        k : number of candidates to build
        ts_names : set of keys in ``active_numeric`` that are time-series modalities
        tab_names : set of keys in ``active_numeric`` that are named tabular modalities
        """
        ts_names = set(ts_names or [])
        tab_names = set(tab_names or [])

        n_per = {role: len(idxs) for role, (idxs, _) in active_numeric.items()}
        for name, (idxs, _) in (active_texts or {}).items():
            n_per[("text", name)] = len(idxs)
        for name, (idxs, _) in (active_images or {}).items():
            n_per[("image", name)] = len(idxs)
        if not n_per:
            return []
        k_eff = min(k, *n_per.values())
        if k_eff <= 0:
            return []

        results = []
        for i in range(1, k_eff + 1):
            use_n = i
            candidate: dict = {}
            source_indices: dict = {"tabular": {}, "ts": {}, "text": {}, "image": {}}

            for role, (idxs, train_arr) in active_numeric.items():
                idx_arr = np.asarray(idxs[:use_n], dtype=int)
                vals = np.asarray(train_arr[idx_arr])
                med = vals[0] if use_n == 1 else np.median(vals, axis=0)
                candidate[role] = np.asarray(med, dtype=float)
                if role in ts_names:
                    source_indices["ts"][role] = int(idxs[i - 1])
                else:
                    source_indices["tabular"][role] = int(idxs[i - 1])

            text_vals = {}
            text_inputs = {}
            for text_name, (text_idxs, train_text_raw) in (active_texts or {}).items():
                train_text_np = np.asarray(train_text_raw, dtype=object)
                anchor = int(text_idxs[i - 1])
                text_vals[text_name] = str(train_text_np[anchor])
                text_inputs[text_name] = train_text_np[anchor]
                source_indices["text"][text_name] = anchor

            image_vals = {}
            image_inputs = {}
            for image_name, (img_idxs, train_img) in (active_images or {}).items():
                img_anchor = int(img_idxs[i - 1])
                image_vals[image_name] = train_img[img_anchor]
                image_inputs[image_name] = train_img[img_anchor]
                source_indices["image"][image_name] = img_anchor

            ts_dict = {role: candidate[role] for role in candidate if role in ts_names}
            tab_dict = {role: candidate[role] for role in candidate if role in tab_names}
            text_val = text_vals.get(primary_text_name)
            text_input = text_inputs.get(primary_text_name)
            img_val = image_vals.get(primary_image_name)

            results.append(
                {
                    "static": candidate.get("static"),
                    "ts": ts_dict,
                    "tab": tab_dict,
                    "text": text_val,
                    "text_input": text_input,
                    "texts": text_vals,
                    "text_inputs": text_inputs,
                    "image": img_val,
                    "image_input": img_val,
                    "images": image_vals,
                    "image_inputs": image_inputs,
                    "source_indices": source_indices,
                    "n_neighbors_used": use_n,
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
        """Generate Frankenstein counterfactuals.

        Parameters
        ----------
        precomputed : optional dict with any of the following keys to skip the
            corresponding internal unimodal search::

                {
                    "tabular":   (indices_dict, distances_dict),
                    <ts_name>:   (indices_dict, distances_dict),
                    ("text", name):  (indices_dict, distances_dict),
                    "train_text_str": {name: [...]},
                    ("image", name): (indices_dict, distances_dict),
                }

            Missing keys are computed on-the-fly.
        """
        k = k if k is not None else self.k
        pc = precomputed or {}
        avail = dataset.available_modalities

        # --- static NN (only when tabular modality is present) ---
        idx_static = None
        if "tabular" in avail:
            if "tabular" in pc:
                idx_static, _ = pc["tabular"]
            else:
                idx_static, _ = find_k_closest_static(
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
        ts_indices: Dict[str, dict] = {}
        ts_names = list((dataset.X_train_ts or {}).keys())
        for ts_name in ts_names:
            if ts_name in avail:
                if ts_name in pc:
                    ts_indices[ts_name], _ = pc[ts_name]
                else:
                    idx_ts, _ = find_k_closest_ts(
                        X_train_ts=dataset.X_train_ts[ts_name],
                        y_train=dataset.y_train,
                        X_test_ts=dataset.X_test_ts[ts_name],
                        selected_test_indices=[sample_idx],
                        target_value=target_value,
                        k=self.k_search,
                        distance_fn=self.ts_dist_fn,
                        return_indices=True,
                    )
                    ts_indices[ts_name] = idx_ts

        # --- per-named-tabular NN ---
        tab_indices: Dict[str, dict] = {}
        tab_names_list = list((dataset.X_train_tab or {}).keys())
        for tab_name in tab_names_list:
            if tab_name in avail:
                if tab_name in pc:
                    tab_indices[tab_name], _ = pc[tab_name]
                else:
                    idx_tab, _ = find_k_closest_static(
                        X_train_static=dataset.X_train_tab[tab_name],
                        y_train=dataset.y_train,
                        X_test_static=dataset.X_test_tab[tab_name],
                        selected_test_indices=[sample_idx],
                        target_value=target_value,
                        k=self.k_search,
                        distance_fn=self.static_dist_fn,
                        return_indices=True,
                    )
                    tab_indices[tab_name] = idx_tab

        # --- text NN (skipped when not in dataset) ---
        text_indices: Dict[str, dict] = {}
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
                text_indices[text_name], _ = pc[cache_key]
            else:
                ts_list = list((dataset.X_train_ts or {}).values())
                train_ts1 = ts_list[0] if len(ts_list) > 0 else None
                train_ts2 = ts_list[1] if len(ts_list) > 1 else None
                ts_test_list = list((dataset.X_test_ts or {}).values())
                test_ts1 = ts_test_list[0] if len(ts_test_list) > 0 else None
                test_ts2 = ts_test_list[1] if len(ts_test_list) > 1 else None
                _, idx_text, _ = find_k_closest_text(
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
                text_indices[text_name] = idx_text

        # --- image NN (skipped when not in dataset) ---
        image_indices: Dict[str, dict] = {}
        for image_name in dataset.image_names():
            cache_key = ("image", image_name)
            train_img = dataset.get_image_branch(image_name, split="train")
            test_img = dataset.get_image_branch(image_name, split="test")
            if train_img is None or test_img is None:
                continue
            if cache_key in pc:
                image_indices[image_name], _ = pc[cache_key]
            else:
                idx_image, _ = find_k_closest_image(
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
                image_indices[image_name] = idx_image

        # --- build active_numeric, active_text, active_image ---
        active_numeric: dict = {}
        if idx_static is not None:
            active_numeric["static"] = (
                np.asarray(idx_static.get(int(sample_idx), []), dtype=int),
                dataset.X_train_static,
            )
        for ts_name, idx_ts in ts_indices.items():
            active_numeric[ts_name] = (
                np.asarray(idx_ts.get(int(sample_idx), []), dtype=int),
                dataset.X_train_ts[ts_name],
            )
        for tab_name, idx_tab in tab_indices.items():
            active_numeric[tab_name] = (
                np.asarray(idx_tab.get(int(sample_idx), []), dtype=int),
                dataset.X_train_tab[tab_name],
            )

        active_texts = {
            text_name: (
                np.asarray(idx_text.get(int(sample_idx), []), dtype=int),
                dataset.get_text_branch(text_name, split="train"),
            )
            for text_name, idx_text in text_indices.items()
        }

        active_images = {
            image_name: (
                np.asarray(idx_image.get(int(sample_idx), []), dtype=int),
                dataset.get_image_branch(image_name, split="train"),
            )
            for image_name, idx_image in image_indices.items()
        }

        if not active_numeric and not active_texts and not active_images:
            print("[ModalityWisePrototypeSynthesis] no modalities available for this dataset — returning empty candidates.")
            return []

        return self._frankenstein_partial(
            active_numeric,
            active_texts,
            active_images,
            k,
            ts_names=set(ts_names),
            tab_names=set(tab_indices.keys()),
            primary_text_name=dataset.primary_text_name,
            primary_image_name=dataset.primary_image_name,
        )


# Short alias matching the paper's abbreviation (MPS).
MPS = ModalityWisePrototypeSynthesis
