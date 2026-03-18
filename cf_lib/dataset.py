from __future__ import annotations

from dataclasses import dataclass, field
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

    X_train_img : array-like of length n_train or None
        Raw images (list of PIL Images, (n, H, W, C) ndarray) **or**
        pre-computed embeddings ((n, D) float ndarray).
    X_test_img  : same format as ``X_train_img`` but for the test split.

    y_test       : np.ndarray, shape (n_test,)             or None
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

    # --- optional text ---
    X_train_text: Optional[Any] = None   # strings, PIDs, or any object array
    X_test_text: Optional[Any] = None

    # --- optional image ---
    X_train_img: Optional[Any] = None   # PIL list / (n,H,W,C) array / (n,D) embeddings
    X_test_img: Optional[Any] = None

    # --- optional test labels ---
    y_test: Optional[np.ndarray] = None

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
        if self.X_train_tab is not None:
            mods.update(self.X_train_tab.keys())
        if self.X_train_text is not None:
            mods.add("text")
        if self.X_train_img is not None:
            mods.add("image")
        return mods
