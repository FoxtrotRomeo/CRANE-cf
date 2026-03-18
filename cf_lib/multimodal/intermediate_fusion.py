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

    Model input conventions (detected automatically from ``len(model.inputs)``):
    - **1 input**  — static features only.
    - **4 inputs** — ``[meds, labs, static, text]``.
    - **5 inputs** — ``[meds, labs, static, text, image]``.
      Requires ``img_embed_fn`` or a non-``"precomputed"`` ``image_encoder``
      when the dataset contains raw images.

    For models with a different input layout, pass ``precomputed_train_latent``
    and ``precomputed_test_latent`` directly to bypass the Keras prediction.

    Parameters
    ----------
    k            : default number of candidates
    ts1_name     : key in ``dataset.X_train_ts`` to use as the first ts input
                   (defaults to the first key in insertion order)
    ts2_name     : key in ``dataset.X_train_ts`` to use as the second ts input
                   (defaults to the second key, or None)
    img_embed_fn : optional callable ``(images) -> np.ndarray (n, D)``; used
                   to encode images before passing them to the model.  When
                   ``None`` the raw ``X_train_img`` / ``X_test_img`` arrays are
                   passed directly (appropriate when they are already embeddings
                   or when the model accepts raw pixel arrays).
    image_encoder: backbone string used when ``img_embed_fn`` is None and
                   image encoding is needed — ``"precomputed"`` (default,
                   pass arrays as-is), ``"resnet50"``, ``"efficientnet_b0"``,
                   ``"clip_vit_b32"``, or ``"custom"``.
    img_device   : torch device string for image encoding
    img_batch_size: images per forward pass
    distance_fn  : latent-space distance callable (default: sklearn euclidean_distances)
    distance_metric : optional metric string; takes precedence over ``distance_fn``.
    """

    def __init__(
        self,
        k: int = 5,
        ts1_name: Optional[str] = None,
        ts2_name: Optional[str] = None,
        img_embed_fn: Optional[Callable] = None,
        image_encoder: str = "precomputed",
        img_device: Optional[str] = None,
        img_batch_size: int = 32,
        distance_fn: Callable = euclidean_distances,
        distance_metric: Optional[str] = None,
    ):
        self.k = k
        self.ts1_name = ts1_name
        self.ts2_name = ts2_name
        self.img_embed_fn = img_embed_fn
        self.image_encoder = image_encoder
        self.img_device = img_device
        self.img_batch_size = img_batch_size
        self.distance_fn = distance_fn
        self.distance_metric = distance_metric

    def _resolve_ts(self, dataset, split: str = "train"):
        """Return (arr1, arr2) for the configured ts names, or None when absent."""
        ts = (dataset.X_train_ts if split == "train" else dataset.X_test_ts) or {}
        keys = list(ts.keys())
        name1 = self.ts1_name or (keys[0] if len(keys) > 0 else None)
        name2 = self.ts2_name or (keys[1] if len(keys) > 1 else None)
        return ts.get(name1), ts.get(name2)

    def _encode_images(self, images):
        """Return encoded images as a numpy array, or pass through if precomputed."""
        if images is None:
            return None
        if self.img_embed_fn is not None:
            import numpy as np
            return np.asarray(self.img_embed_fn(images), dtype=float)
        enc = str(self.image_encoder).strip().lower() if self.image_encoder else "precomputed"
        if enc == "precomputed":
            import numpy as np
            return np.asarray(images, dtype=float)
        from counterfactual_helpers import _build_image_representations
        emb, _ = _build_image_representations(
            images, images,
            image_encoder=enc,
            embed_fn=None,
            device=self.img_device,
            batch_size=self.img_batch_size,
        )
        return emb

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

        # Encode images when the dataset has an image modality.
        train_img = self._encode_images(dataset.X_train_img) if "image" in dataset.available_modalities else None
        test_img = self._encode_images(dataset.X_test_img) if "image" in dataset.available_modalities else None

        # find_k_closest_latent_model requires a non-None X_train; fall back to
        # a zero placeholder when the static modality is absent.
        import numpy as np
        n_train = len(dataset.y_train)
        n_test = (
            dataset.X_test_static.shape[0] if dataset.X_test_static is not None
            else (len(dataset.X_test_img) if dataset.X_test_img is not None
                  else len(dataset.X_test_text) if dataset.X_test_text is not None
                  else 0)
        )
        X_train = dataset.X_train_static if dataset.X_train_static is not None else np.zeros((n_train, 1))
        X_test = dataset.X_test_static if dataset.X_test_static is not None else np.zeros((n_test, 1))

        neighbors, _, _ = find_k_closest_latent_model(
            X_train=X_train,
            y_train=dataset.y_train,
            X_test=X_test,
            selected_test_indices=[sample_idx],
            model=model,
            X_train_meds=train_ts1,
            X_train_labs=train_ts2,
            X_train_text=dataset.X_train_text,
            X_test_meds=test_ts1,
            X_test_labs=test_ts2,
            X_test_text=dataset.X_test_text,
            X_train_img=train_img,
            X_test_img=test_img,
            target_value=target_value,
            k=k,
            distance_fn=self.distance_fn,
            distance_metric=self.distance_metric,
            return_latents=True,
        )
        return neighbors.get(int(sample_idx), [])
