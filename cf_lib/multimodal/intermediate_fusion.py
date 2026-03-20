"""Intermediate-Fusion nearest-neighbour counterfactual generator."""
from __future__ import annotations

import inspect
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from sklearn.metrics.pairwise import euclidean_distances

from counterfactual_helpers import find_k_closest_latent, find_k_closest_latent_model
from cf_lib.base import CounterfactualGenerator


class IntermediateFusionNN(CounterfactualGenerator):
    """Nearest-neighbour counterfactuals in the classifier's latent space.

    Preferred usage is to provide either precomputed train/test latent
    matrices or a ``latent_fn`` that knows how to build them from the dataset.
    A legacy Keras-specific fallback path is retained for older callers.

    Parameters
    ----------
    k            : default number of candidates
    latent_fn    : optional callable that returns latent representations for
                   a given split. Preferred signature:
                   ``latent_fn(dataset, split=\"train\", model=None) -> ndarray``.
    precomputed_train_latent / precomputed_test_latent
                 : explicit latent matrices used directly when provided.
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
        latent_fn: Optional[Callable] = None,
        precomputed_train_latent=None,
        precomputed_test_latent=None,
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
        self.latent_fn = latent_fn
        self.precomputed_train_latent = precomputed_train_latent
        self.precomputed_test_latent = precomputed_test_latent
        self.ts1_name = ts1_name
        self.ts2_name = ts2_name
        self.img_embed_fn = img_embed_fn
        self.image_encoder = image_encoder
        self.img_device = img_device
        self.img_batch_size = img_batch_size
        self.distance_fn = distance_fn
        self.distance_metric = distance_metric

    @staticmethod
    def _materialize(
        indices_dict: dict,
        sample_idx: int,
        dataset,
        distance_metric_label: str,
    ) -> List[Dict[str, Any]]:
        idxs = np.asarray(indices_dict.get(int(sample_idx), []), dtype=int)
        n_avail = idxs.size
        if n_avail == 0:
            return []

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
        train_texts = {
            name: np.asarray(arr, dtype=object).reshape(-1)
            for name, arr in dataset.text_modalities("train").items()
        }
        train_images = dataset.image_modalities("train")

        results = []
        for i in range(1, n_avail + 1):
            use_n = min(2 * i - 1, n_avail)
            subset = idxs[:use_n]
            anchor = int(idxs[i - 1])

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
            text_vals = {name: str(arr[anchor]) for name, arr in train_texts.items()}
            text_inputs = {name: arr[anchor] for name, arr in train_texts.items()}
            image_vals = {name: arr[anchor] for name, arr in train_images.items()}
            image_inputs = dict(image_vals)

            results.append(
                {
                    "static": static_med,
                    "ts": ts_meds,
                    "tab": tab_meds,
                    "text": text_vals.get(dataset.primary_text_name),
                    "text_input": text_inputs.get(dataset.primary_text_name),
                    "texts": text_vals,
                    "text_inputs": text_inputs,
                    "image": image_vals.get(dataset.primary_image_name),
                    "image_input": image_vals.get(dataset.primary_image_name),
                    "images": image_vals,
                    "image_inputs": image_inputs,
                    "source_train_idx": anchor,
                    "source_indices": {
                        "tabular": (
                            {name: anchor for name in train_tab}
                            | ({dataset.primary_tabular_name: anchor} if train_static is not None else {})
                        ),
                        "ts": {name: anchor for name in train_ts},
                        "text": {name: anchor for name in train_texts},
                        "image": {name: anchor for name in train_images},
                    },
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

    def _call_latent_fn(self, dataset, split: str, model=None):
        if self.latent_fn is None:
            return None
        fn = self.latent_fn
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            if "dataset" in sig.parameters:
                kwargs["dataset"] = dataset
            if "split" in sig.parameters:
                kwargs["split"] = split
            if "model" in sig.parameters:
                kwargs["model"] = model
            if kwargs:
                return fn(**kwargs)
        except Exception:
            pass
        try:
            return fn(dataset, split, model)
        except TypeError:
            return fn(dataset, split)

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
        k = k if k is not None else self.k
        train_latent = self.precomputed_train_latent
        test_latent = self.precomputed_test_latent

        if train_latent is None and self.latent_fn is not None:
            train_latent = self._call_latent_fn(dataset, "train", model=model)
        if test_latent is None and self.latent_fn is not None:
            test_latent = self._call_latent_fn(dataset, "test", model=model)

        if train_latent is not None and test_latent is not None:
            indices_dict, _ = find_k_closest_latent(
                X_train_latent=train_latent,
                y_train=dataset.y_train,
                X_test_latent=test_latent,
                selected_test_indices=[sample_idx],
                target_value=target_value,
                k=k,
                distance_fn=self.distance_fn,
                distance_metric=self.distance_metric,
            )
            return self._materialize(
                indices_dict,
                sample_idx,
                dataset,
                distance_metric_label=self._resolve_distance_metric_label(),
            )

        if model is None:
            raise ValueError(
                "IntermediateFusionNN requires either "
                "(precomputed_train_latent and precomputed_test_latent), "
                "or latent_fn, or a legacy-compatible trained model."
            )

        train_ts1, train_ts2 = self._resolve_ts(dataset, "train")
        test_ts1, test_ts2 = self._resolve_ts(dataset, "test")

        train_img = (
            self._encode_images(dataset.X_train_img)
            if "image" in dataset.available_modalities else None
        )
        test_img = (
            self._encode_images(dataset.X_test_img)
            if "image" in dataset.available_modalities else None
        )

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

        _, _, indices_dict = find_k_closest_latent_model(
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
            return_indices=True,
        )
        return self._materialize(
            indices_dict,
            sample_idx,
            dataset,
            distance_metric_label=self._resolve_distance_metric_label(),
        )

    def generate_batch(
        self,
        dataset,
        sample_indices,
        model=None,
        target_value: int = 0,
        k: Optional[int] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        k = k if k is not None else self.k
        selected = [int(idx) for idx in sample_indices]
        train_latent = self.precomputed_train_latent
        test_latent = self.precomputed_test_latent

        if train_latent is None and self.latent_fn is not None:
            train_latent = self._call_latent_fn(dataset, "train", model=model)
        if test_latent is None and self.latent_fn is not None:
            test_latent = self._call_latent_fn(dataset, "test", model=model)

        if train_latent is not None and test_latent is not None:
            indices_dict, _ = find_k_closest_latent(
                X_train_latent=train_latent,
                y_train=dataset.y_train,
                X_test_latent=test_latent,
                selected_test_indices=selected,
                target_value=target_value,
                k=k,
                distance_fn=self.distance_fn,
                distance_metric=self.distance_metric,
            )
            return {
                int(idx): self._materialize(
                    indices_dict,
                    int(idx),
                    dataset,
                    distance_metric_label=self._resolve_distance_metric_label(),
                )
                for idx in selected
            }

        if model is None:
            raise ValueError(
                "IntermediateFusionNN requires either "
                "(precomputed_train_latent and precomputed_test_latent), "
                "or latent_fn, or a legacy-compatible trained model."
            )

        train_ts1, train_ts2 = self._resolve_ts(dataset, "train")
        test_ts1, test_ts2 = self._resolve_ts(dataset, "test")

        train_img = (
            self._encode_images(dataset.X_train_img)
            if "image" in dataset.available_modalities else None
        )
        test_img = (
            self._encode_images(dataset.X_test_img)
            if "image" in dataset.available_modalities else None
        )

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

        _, _, indices_dict = find_k_closest_latent_model(
            X_train=X_train,
            y_train=dataset.y_train,
            X_test=X_test,
            selected_test_indices=selected,
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
            return_indices=True,
        )
        return {
            int(idx): self._materialize(
                indices_dict,
                int(idx),
                dataset,
                distance_metric_label=self._resolve_distance_metric_label(),
            )
            for idx in selected
        }
