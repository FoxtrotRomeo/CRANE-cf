"""Image nearest-neighbour counterfactual generator."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from cf_lib.counterfactual_helpers import find_k_closest_image
from cf_lib.base import CounterfactualGenerator


class ImageNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in configurable image embedding spaces.

    Accepts either **pre-computed embeddings** or **raw images** (PIL / numpy
    arrays) that are encoded on-the-fly with a built-in or custom backbone.

    Auto-detection rule (when ``image_encoder`` is ``"precomputed"``):
    - 2-D float ndarray or list of 1-D vectors → treated as embeddings.
    - 3-D/4-D ndarray, list of PIL Images, or list of (H, W, C) arrays
      → treated as raw images (requires a non-precomputed encoder).

    Supported backbones (``image_encoder`` parameter):
    - ``"precomputed"`` (default) — embeddings passed as-is.
    - ``"resnet50"``              — torchvision ResNet-50 (2 048-d).
    - ``"efficientnet_b0"``       — torchvision EfficientNet-B0 (1 280-d).
    - ``"clip_vit_b32"``          — OpenAI CLIP ViT-B/32 (512-d).
    - ``"custom"``                — user provides ``embed_fn``.

    Numeric modalities (static + all time-series / named-tabular blocks)
    are built as odd-k progressive medians.  The image modality uses
    rank-matched selection: candidate *i* draws its image from the *i*-th
    nearest neighbor.

    Parameters
    ----------
    image_name           : key in ``dataset.X_train_images`` / ``dataset.X_test_images``,
                           or ``None`` to use the resolved primary / sole image branch.
    k                    : default number of candidates
    image_encoder        : backbone identifier (see above)
    image_distance_metric: vector metric — ``"cosine"`` (default),
                           ``"euclidean"``, ``"manhattan"``, ``"hamming"``,
                           or any sklearn-compatible metric string
    image_distance_fn    : optional pairwise-distance callable; takes
                           precedence over ``image_distance_metric``
    embed_fn             : optional ``callable(images) -> np.ndarray (n, D)``;
                           takes precedence over ``image_encoder``
    device               : torch device string (e.g. ``"cuda"`` or ``"cpu"``).
                           Defaults to CUDA when available, else CPU.
    batch_size           : images per forward pass when encoding raw images
    """

    def __init__(
        self,
        image_name: Optional[str] = None,
        k: int = 20,
        image_encoder: str = "precomputed",
        image_distance_metric: str = "cosine",
        image_distance_fn: Optional[Callable] = None,
        embed_fn: Optional[Callable] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.image_name = image_name
        self.k = k
        self.image_encoder = image_encoder
        self.image_distance_metric = image_distance_metric
        self.image_distance_fn = image_distance_fn
        self.embed_fn = embed_fn
        self.device = device
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    @staticmethod
    def _materialize(
        indices_dict: dict,
        sample_idx: int,
        dataset,
        train_img,
        image_name: str,
        distance_metric_label: str,
        image_encoder_label: str,
    ) -> List[Dict[str, Any]]:
        """Build progressive-median candidate dicts from neighbor indices.

        Image uses rank-matched selection (candidate *i* → neighbor *i*).
        Numeric modalities (static, TS, named-tabular) use progressive medians.
        """
        idxs = np.asarray(indices_dict.get(int(sample_idx), []), dtype=int)
        n_avail = idxs.size
        if n_avail == 0:
            return []

        # Prepare train arrays; skip absent modalities gracefully.
        train_static = (
            np.asarray(dataset.X_train_static, dtype=float)
            if dataset.X_train_static is not None else None
        )
        train_ts = {
            name: np.asarray(arr, dtype=float)
            for name, arr in (dataset.X_train_ts or {}).items()
        }
        train_tab = {
            name: np.asarray(arr, dtype=float)
            for name, arr in (dataset.X_train_tab or {}).items()
        }
        results = []
        for i in range(1, n_avail + 1):
            use_n = min(2 * i - 1, n_avail)
            subset = idxs[:use_n]
            anchor = int(idxs[i - 1])   # rank-matched: candidate i → neighbor i

            static_med = (
                np.asarray(np.median(train_static[subset], axis=0), dtype=float)
                if train_static is not None else None
            )
            ts_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_ts.items()
            }
            tab_meds = {
                name: np.asarray(np.median(arr[subset], axis=0), dtype=float)
                for name, arr in train_tab.items()
            }

            # Rank-matched image retrieval.
            img_val = train_img[anchor]

            results.append(
                {
                    "static": static_med,
                    "ts": ts_meds,
                    "tab": tab_meds,
                    "image": img_val,
                    "image_input": img_val,
                    "images": {image_name: img_val},
                    "image_inputs": {image_name: img_val},
                    "image_branch": image_name,
                    "source_train_idx": anchor,
                    "n_neighbors_used": int(use_n),
                    "distance_metric": distance_metric_label,
                    "image_encoder": image_encoder_label,
                }
            )
        return results

    # ------------------------------------------------------------------
    def _resolve_distance_metric_label(self) -> str:
        if self.image_distance_metric is not None:
            return str(self.image_distance_metric)
        if self.image_distance_fn is None:
            return "cosine"
        return getattr(
            self.image_distance_fn,
            "__name__",
            self.image_distance_fn.__class__.__name__,
        )

    def _resolve_encoder_label(self) -> str:
        if self.embed_fn is not None:
            return "custom"
        return str(self.image_encoder) if self.image_encoder is not None else "precomputed"

    # ------------------------------------------------------------------
    def generate_raw(
        self,
        dataset,
        sample_idx: int,
        target_value: int = 0,
        k: Optional[int] = None,
    ):
        """Run the image NN search.

        Returns ``(candidates, indices_dict, distances_dict)``.
        """
        image_name = dataset.resolve_image_branch_name(self.image_name)
        train_img = dataset.get_image_branch(image_name, split="train")
        test_img = dataset.get_image_branch(image_name, split="test")
        if train_img is None or test_img is None:
            print("[ImageNN] image modality not present in dataset — returning empty candidates.")
            return [], {}, {}

        k = k if k is not None else self.k

        indices_dict, distances_dict = find_k_closest_image(
            X_train_img=train_img,
            y_train=dataset.y_train,
            X_test_img=test_img,
            selected_test_indices=[sample_idx],
            target_value=target_value,
            k=k,
            image_encoder=self.image_encoder,
            image_distance_metric=self.image_distance_metric,
            image_distance_fn=self.image_distance_fn,
            embed_fn=self.embed_fn,
            device=self.device,
            batch_size=self.batch_size,
        )

        candidates = self._materialize(
            indices_dict,
            sample_idx,
            dataset,
            train_img,
            image_name,
            distance_metric_label=self._resolve_distance_metric_label(),
            image_encoder_label=self._resolve_encoder_label(),
        )
        return candidates, indices_dict, distances_dict

    def generate_raw_batch(
        self,
        dataset,
        sample_indices,
        target_value: int = 0,
        k: Optional[int] = None,
    ):
        """Run the image NN search once for many samples.

        Returns ``(candidates_by_sample, indices_dict, distances_dict)``.
        """
        image_name = dataset.resolve_image_branch_name(self.image_name)
        train_img = dataset.get_image_branch(image_name, split="train")
        test_img = dataset.get_image_branch(image_name, split="test")
        selected = [int(idx) for idx in sample_indices]
        if train_img is None or test_img is None:
            print("[ImageNN] image modality not present in dataset — returning empty candidates.")
            empty = {int(idx): [] for idx in selected}
            return empty, {}, {}

        k = k if k is not None else self.k

        indices_dict, distances_dict = find_k_closest_image(
            X_train_img=train_img,
            y_train=dataset.y_train,
            X_test_img=test_img,
            selected_test_indices=selected,
            target_value=target_value,
            k=k,
            image_encoder=self.image_encoder,
            image_distance_metric=self.image_distance_metric,
            image_distance_fn=self.image_distance_fn,
            embed_fn=self.embed_fn,
            device=self.device,
            batch_size=self.batch_size,
        )

        candidates_by_sample = {
            int(idx): self._materialize(
                indices_dict,
                int(idx),
                dataset,
                train_img,
                image_name,
                distance_metric_label=self._resolve_distance_metric_label(),
                image_encoder_label=self._resolve_encoder_label(),
            )
            for idx in selected
        }
        return candidates_by_sample, indices_dict, distances_dict

    # ------------------------------------------------------------------
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

    def generate_batch(
        self,
        dataset,
        sample_indices,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        candidates_by_sample, _, _ = self.generate_raw_batch(
            dataset,
            sample_indices,
            target_value=target_value,
            k=k,
        )
        return candidates_by_sample
