from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class MultimodalDataset:
    """Container for all modality arrays used by the counterfactual generators.

    Only labels (``y_train``) are strictly required.  All data arrays are
    optional and default to ``None``.  Generators that require a missing
    modality will return an empty candidate list and print a warning, so you
    can safely register only the generators that match the modalities present
    in your dataset.

    Required parameters
    -------------------
    y_train : np.ndarray, shape (n_train,) — integer class labels

    Optional parameters
    -------------------
    X_train_static : np.ndarray, shape (n_train, D_static) or None
    X_test_static  : np.ndarray, shape (n_test,  D_static) or None

    X_train_ts : dict[str, np.ndarray] or None
        Named time-series modalities.  Each value has shape
        ``(n_train, T, D)`` where T and D may differ per key.
        Example: ``{"ts1": arr1, "ts2": arr2}``
    X_test_ts  : dict[str, np.ndarray] or None
        Same keys as ``X_train_ts`` but for the test split.

    X_train_tab : dict[str, np.ndarray] or None
        Named tabular modalities (2-D arrays of shape ``(n_train, D)``).
        Example: ``{"tab1": arr1, "tab2": arr2}``
    X_test_tab  : dict[str, np.ndarray] or None
        Same keys as ``X_train_tab`` but for the test split.

    X_train_text : array-like of length n_train (raw strings / PIDs) or None
    X_test_text  : array-like of length n_test             or None

    X_train_texts : dict[str, array-like] or None
        Named text branches. Example:
        ``{"description": desc_train, "requirements": req_train}``
    X_test_texts  : dict[str, array-like] or None
        Same keys as ``X_train_texts`` but for the test split.

    X_train_img : array-like of length n_train or None
        Raw images (list of PIL Images, (n, H, W, C) ndarray) **or**
        pre-computed embeddings ((n, D) float ndarray).
    X_test_img  : same format as ``X_train_img`` but for the test split.

    X_train_images : dict[str, array-like] or None
        Named image branches. Example:
        ``{"cxr": cxr_train, "ct": ct_train}``
    X_test_images  : dict[str, array-like] or None
        Same keys as ``X_train_images`` but for the test split.

    y_test       : np.ndarray, shape (n_test,)             or None

    Notes
    -----
    Named branches are treated as first-class modalities. A primary branch is
    only resolved when it is explicitly named or when exactly one branch exists
    for that modality. When multiple named branches are present and no primary
    is resolved, the legacy flat aliases (for example ``X_train_text``) remain
    ``None`` and callers must specify branch names explicitly.
    """

    # --- required ---
    y_train: np.ndarray

    # --- optional static ---
    X_train_static: Optional[np.ndarray] = None
    X_test_static: Optional[np.ndarray] = None

    # --- optional time-series ---
    X_train_ts: Optional[Dict[str, np.ndarray]] = None
    X_test_ts: Optional[Dict[str, np.ndarray]] = None

    # --- optional named tabular ---
    X_train_tab: Optional[Dict[str, np.ndarray]] = None
    X_test_tab: Optional[Dict[str, np.ndarray]] = None
    X_train_tabular: Optional[Dict[str, np.ndarray]] = None
    X_test_tabular: Optional[Dict[str, np.ndarray]] = None

    # --- optional text ---
    X_train_text: Optional[Any] = None   # strings, PIDs, or any object array
    X_test_text: Optional[Any] = None
    X_train_texts: Optional[Dict[str, Any]] = None
    X_test_texts: Optional[Dict[str, Any]] = None

    # --- optional image ---
    X_train_img: Optional[Any] = None   # PIL list / (n,H,W,C) array / (n,D) embeddings
    X_test_img: Optional[Any] = None
    X_train_images: Optional[Dict[str, Any]] = None
    X_test_images: Optional[Dict[str, Any]] = None

    # --- optional primary branch names for compatibility ---
    primary_tabular_name: Optional[str] = "__primary__"
    primary_text_name: Optional[str] = "__primary__"
    primary_image_name: Optional[str] = "__primary__"

    # --- optional test labels ---
    y_test: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        configured_primary_tab = self.primary_tabular_name
        configured_primary_text = self.primary_text_name
        configured_primary_image = self.primary_image_name

        self.X_train_tabular = self._merge_tabular_dicts(
            primary=self.X_train_static,
            named=self.X_train_tab,
            explicit=self.X_train_tabular,
            primary_name=configured_primary_tab,
        )
        self.X_test_tabular = self._merge_tabular_dicts(
            primary=self.X_test_static,
            named=self.X_test_tab,
            explicit=self.X_test_tabular,
            primary_name=configured_primary_tab,
        )
        self.X_train_texts = self._merge_named_branch_dict(
            legacy_primary=self.X_train_text,
            named=self.X_train_texts,
            primary_name=configured_primary_text,
        )
        self.X_test_texts = self._merge_named_branch_dict(
            legacy_primary=self.X_test_text,
            named=self.X_test_texts,
            primary_name=configured_primary_text,
        )
        self.X_train_images = self._merge_named_branch_dict(
            legacy_primary=self.X_train_img,
            named=self.X_train_images,
            primary_name=configured_primary_image,
        )
        self.X_test_images = self._merge_named_branch_dict(
            legacy_primary=self.X_test_img,
            named=self.X_test_images,
            primary_name=configured_primary_image,
        )

        self.primary_tabular_name = self._resolve_primary_name(
            configured_primary_tab,
            self.X_train_tabular,
            modality_label="tabular",
        )
        self.primary_text_name = self._resolve_primary_name(
            configured_primary_text,
            self.X_train_texts,
            modality_label="text",
        )
        self.primary_image_name = self._resolve_primary_name(
            configured_primary_image,
            self.X_train_images,
            modality_label="image",
        )

        self.X_train_static = (
            self.get_tabular_branch(self.primary_tabular_name, split="train")
            if self.primary_tabular_name is not None
            else None
        )
        self.X_test_static = (
            self.get_tabular_branch(self.primary_tabular_name, split="test")
            if self.primary_tabular_name is not None
            else None
        )
        self.X_train_tab = {
            name: arr
            for name, arr in (self.X_train_tabular or {}).items()
            if self.primary_tabular_name is None or name != self.primary_tabular_name
        } or None
        self.X_test_tab = {
            name: arr
            for name, arr in (self.X_test_tabular or {}).items()
            if self.primary_tabular_name is None or name != self.primary_tabular_name
        } or None
        self.X_train_text = (
            self.get_text_branch(self.primary_text_name, split="train")
            if self.primary_text_name is not None
            else None
        )
        self.X_test_text = (
            self.get_text_branch(self.primary_text_name, split="test")
            if self.primary_text_name is not None
            else None
        )
        self.X_train_img = (
            self.get_image_branch(self.primary_image_name, split="train")
            if self.primary_image_name is not None
            else None
        )
        self.X_test_img = (
            self.get_image_branch(self.primary_image_name, split="test")
            if self.primary_image_name is not None
            else None
        )

    @staticmethod
    def _merge_named_branch_dict(
        *,
        legacy_primary: Optional[Any],
        named: Optional[Dict[str, Any]],
        primary_name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        out: Dict[str, Any] = {}
        if named:
            out.update(named)
        if legacy_primary is not None:
            branch_name = primary_name or "__primary__"
            existing = out.get(branch_name)
            if existing is not None and existing is not legacy_primary:
                raise ValueError(
                    f"Conflicting values provided for primary branch '{branch_name}'."
                )
            out[branch_name] = legacy_primary
        return out or None

    @staticmethod
    def _merge_tabular_dicts(
        *,
        primary: Optional[np.ndarray],
        named: Optional[Dict[str, np.ndarray]],
        explicit: Optional[Dict[str, np.ndarray]],
        primary_name: Optional[str],
    ) -> Optional[Dict[str, np.ndarray]]:
        out: Dict[str, np.ndarray] = {}
        if named:
            out.update(named)
        if explicit:
            for key, value in explicit.items():
                if key in out and out[key] is not value:
                    raise ValueError(f"Conflicting values provided for tabular branch '{key}'.")
                out[key] = value
        if primary is not None:
            branch_name = primary_name or "__primary__"
            existing = out.get(branch_name)
            if existing is not None and existing is not primary:
                raise ValueError(
                    f"Conflicting values provided for primary tabular branch '{branch_name}'."
                )
            out[branch_name] = primary
        return out or None

    @staticmethod
    def _resolve_primary_name(
        configured_name: Optional[str],
        store: Optional[Dict[str, Any]],
        *,
        modality_label: str,
    ) -> Optional[str]:
        if not store:
            return None
        if configured_name is None:
            if len(store) == 1:
                return next(iter(store))
            return None
        if configured_name in store:
            return configured_name
        if configured_name != "__primary__":
            raise ValueError(
                f"Configured primary {modality_label} branch '{configured_name}' "
                f"was not found. Available branches: {sorted(store)}"
            )
        if len(store) == 1:
            return next(iter(store))
        return None

    @staticmethod
    def _resolve_branch_name(
        requested_name: Optional[str],
        store: Optional[Dict[str, Any]],
        primary_name: Optional[str],
        *,
        modality_label: str,
    ) -> Optional[str]:
        if not store:
            return None
        if requested_name is not None:
            if requested_name not in store:
                raise ValueError(
                    f"{modality_label.capitalize()} branch '{requested_name}' was not found. "
                    f"Available branches: {sorted(store)}"
                )
            return requested_name
        if primary_name is not None:
            return primary_name
        if len(store) == 1:
            return next(iter(store))
        raise ValueError(
            f"Multiple {modality_label} branches are present {sorted(store)}. "
            f"Please specify a branch name explicitly."
        )

    def tabular_modalities(self) -> Dict[str, np.ndarray]:
        return dict(self.X_train_tabular or {})

    def text_modalities(self, split: str = "train") -> Dict[str, Any]:
        if split == "train":
            return dict(self.X_train_texts or {})
        if split == "test":
            return dict(self.X_test_texts or {})
        raise ValueError("split must be 'train' or 'test'.")

    def image_modalities(self, split: str = "train") -> Dict[str, Any]:
        if split == "train":
            return dict(self.X_train_images or {})
        if split == "test":
            return dict(self.X_test_images or {})
        raise ValueError("split must be 'train' or 'test'.")

    def tabular_names(self, *, include_primary: bool = True) -> list[str]:
        names = list((self.X_train_tabular or {}).keys())
        if include_primary:
            return names
        if self.primary_tabular_name is None:
            return names
        return [n for n in names if n != self.primary_tabular_name]

    def ts_names(self) -> list[str]:
        return list((self.X_train_ts or {}).keys())

    def text_names(self) -> list[str]:
        return list((self.X_train_texts or {}).keys())

    def image_names(self) -> list[str]:
        return list((self.X_train_images or {}).keys())

    def resolve_tabular_branch_name(self, name: Optional[str]) -> Optional[str]:
        return self._resolve_branch_name(
            name,
            self.X_train_tabular,
            self.primary_tabular_name,
            modality_label="tabular",
        )

    def resolve_text_branch_name(self, name: Optional[str]) -> Optional[str]:
        return self._resolve_branch_name(
            name,
            self.X_train_texts,
            self.primary_text_name,
            modality_label="text",
        )

    def resolve_image_branch_name(self, name: Optional[str]) -> Optional[str]:
        return self._resolve_branch_name(
            name,
            self.X_train_images,
            self.primary_image_name,
            modality_label="image",
        )

    def get_tabular_branch(self, name: Optional[str], *, split: str = "train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        store = self.X_train_tabular if split == "train" else self.X_test_tabular
        branch_name = self._resolve_branch_name(
            name,
            store,
            self.primary_tabular_name,
            modality_label="tabular",
        )
        return None if store is None else store.get(branch_name)

    def get_text_branch(self, name: Optional[str], *, split: str = "train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        store = self.X_train_texts if split == "train" else self.X_test_texts
        branch_name = self._resolve_branch_name(
            name,
            store,
            self.primary_text_name,
            modality_label="text",
        )
        return None if store is None else store.get(branch_name)

    def get_image_branch(self, name: Optional[str], *, split: str = "train"):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'.")
        store = self.X_train_images if split == "train" else self.X_test_images
        branch_name = self._resolve_branch_name(
            name,
            store,
            self.primary_image_name,
            modality_label="image",
        )
        return None if store is None else store.get(branch_name)

    # ------------------------------------------------------------------
    @property
    def available_modalities(self) -> set:
        """Return the set of modality names present in this dataset.

        ``"tabular"`` is included only when ``X_train_static`` is not ``None``.
        Each key in ``X_train_ts`` and ``X_train_tab`` is included as its own
        modality name.  ``"text"`` and ``"image"`` are added when the
        corresponding arrays are present.
        """
        mods: set = set()
        if self.X_train_static is not None:
            mods.add("tabular")
        if self.X_train_ts is not None:
            mods.update(self.X_train_ts.keys())
        if self.X_train_tabular is not None:
            mods.update(
                k for k in self.X_train_tabular.keys()
                if self.primary_tabular_name is None or k != self.primary_tabular_name
            )
        if self.X_train_texts is not None:
            mods.add("text")
        if self.X_train_images is not None:
            mods.add("image")
        return mods
