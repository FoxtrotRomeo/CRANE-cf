"""Factory templates for ``run_distance_ablation.py``.

You can point the ablation runner to functions in this file, e.g.:

    python run_distance_ablation.py \
      --factory ablation_factory_template:build_dataset_and_model \
      --factory-kwargs-json '{"data_root": "/path/to/fold_0", "ts_modalities": ["labs", "meds"], "load_text": true, "load_image": true, "image_encoder": "resnet50"}'

or for a quick smoke test:

    python run_distance_ablation.py \
      --factory ablation_factory_template:build_synthetic_dataset_and_model \
      --max-combinations 5 \
      --max-samples 3
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from cf_lib import MultimodalDataset


def _load_npy(path: Path):
    try:
        return np.load(path)
    except ValueError as exc:
        if "allow_pickle=False" in str(exc):
            return np.load(path, allow_pickle=True)
        raise


def _maybe_load_model(model_path: Optional[str]):
    if model_path is None:
        return None
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    try:
        import keras
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Could not import keras to load model.") from exc
    return keras.models.load_model(path)


def build_dataset_and_model(
    *,
    data_root: str,
    model_path: Optional[str] = None,
    ts_modalities: Optional[Sequence[str]] = None,
    tab_modalities: Optional[Sequence[str]] = None,
    load_text: bool = True,
    load_image: bool = False,
    image_encoder: str = "precomputed",
) -> Dict[str, Any]:
    """Load dataset/model from a folder layout.

    Required files in ``data_root``:
      - ``y_train.npy``

    Optional files:
      - static:  ``X_train_static.npy``, ``X_test_static.npy``
      - labels:  ``y_test.npy``
      - text:    ``X_train_text.npy``, ``X_test_text.npy`` (when ``load_text=True``)
      - image:   ``X_train_img.npy``, ``X_test_img.npy``  (when ``load_image=True``)
                 Pre-computed embeddings (n, D) or raw pixel arrays.
      - TS modality ``m``: ``X_train_ts_{m}.npy``, ``X_test_ts_{m}.npy``
      - Tab modality ``m``: ``X_train_tab_{m}.npy``, ``X_test_tab_{m}.npy``

    Parameters
    ----------
    image_encoder
        Encoder to use when raw images are provided. Returned in
        ``image_backend_kwargs`` so the ablation runner can forward it to
        ``ImageNN``. One of ``"precomputed"`` (default), ``"resnet50"``,
        ``"efficientnet_b0"``, ``"clip_vit_b32"``, or ``"custom"``.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    y_train = np.asarray(_load_npy(root / "y_train.npy")).reshape(-1)

    # Static (optional)
    X_train_static = None
    X_test_static = None
    static_tr = root / "X_train_static.npy"
    static_te = root / "X_test_static.npy"
    if static_tr.exists() and static_te.exists():
        X_train_static = np.asarray(_load_npy(static_tr), dtype=float)
        X_test_static = np.asarray(_load_npy(static_te), dtype=float)
    else:
        print("[factory] Static files not found, proceeding without static modality.")

    y_test_path = root / "y_test.npy"
    y_test = np.asarray(_load_npy(y_test_path)).reshape(-1) if y_test_path.exists() else None

    X_train_ts = {}
    X_test_ts = {}
    for name in list(ts_modalities or []):
        tr = root / f"X_train_ts_{name}.npy"
        te = root / f"X_test_ts_{name}.npy"
        if tr.exists() and te.exists():
            X_train_ts[name] = np.asarray(_load_npy(tr), dtype=float)
            X_test_ts[name] = np.asarray(_load_npy(te), dtype=float)
        else:
            print(f"[factory] Skipping TS modality '{name}' (missing files).")

    X_train_tab = {}
    X_test_tab = {}
    for name in list(tab_modalities or []):
        tr = root / f"X_train_tab_{name}.npy"
        te = root / f"X_test_tab_{name}.npy"
        if tr.exists() and te.exists():
            X_train_tab[name] = np.asarray(_load_npy(tr), dtype=float)
            X_test_tab[name] = np.asarray(_load_npy(te), dtype=float)
        else:
            print(f"[factory] Skipping tab modality '{name}' (missing files).")

    X_train_text = None
    X_test_text = None
    if load_text:
        tr_text = root / "X_train_text.npy"
        te_text = root / "X_test_text.npy"
        if tr_text.exists() and te_text.exists():
            X_train_text = _load_npy(tr_text)
            X_test_text = _load_npy(te_text)
        else:
            print("[factory] Text files not found, proceeding without text.")

    X_train_img = None
    X_test_img = None
    if load_image:
        tr_img = root / "X_train_img.npy"
        te_img = root / "X_test_img.npy"
        if tr_img.exists() and te_img.exists():
            X_train_img = _load_npy(tr_img)
            X_test_img = _load_npy(te_img)
        else:
            print("[factory] Image files not found, proceeding without image modality.")

    dataset = MultimodalDataset(
        y_train=y_train,
        X_train_static=X_train_static,
        X_test_static=X_test_static,
        X_train_ts=(X_train_ts or None),
        X_test_ts=(X_test_ts or None),
        X_train_tab=(X_train_tab or None),
        X_test_tab=(X_test_tab or None),
        X_train_text=X_train_text,
        X_test_text=X_test_text,
        X_train_img=X_train_img,
        X_test_img=X_test_img,
        y_test=y_test,
    )

    model = _maybe_load_model(model_path)

    # Add text backend objects here if you want to use e5/bert/word2vec in ablations.
    # Example:
    # text_backend_kwargs = {"e5_tokenizer": tokenizer, "e5_model": e5_model, "e5_device": "cuda"}
    text_backend_kwargs: Dict[str, Any] = {}

    # Add image backend objects here if you want to control encoding device / batch size.
    # Example:
    # image_backend_kwargs = {"image_encoder": "resnet50", "device": "cuda", "batch_size": 64}
    image_backend_kwargs: Dict[str, Any] = {"image_encoder": image_encoder}

    return {
        "dataset": dataset,
        "model": model,
        "text_backend_kwargs": text_backend_kwargs,
        "image_backend_kwargs": image_backend_kwargs,
    }


def build_synthetic_dataset_and_model(
    *,
    seed: int = 7,
    n_train: int = 80,
    n_test: int = 20,
    t: int = 12,
    d_static: int = 6,
    d_ts: int = 4,
    d_img: int = 16,
) -> Dict[str, Any]:
    """Return a tiny synthetic dataset for smoke-testing the ablation script.

    Parameters
    ----------
    d_img
        Dimensionality of the synthetic pre-computed image embeddings.
        Set to 0 to omit the image modality entirely.
    """
    rng = np.random.default_rng(int(seed))

    X_train_static = rng.normal(size=(n_train, d_static))
    X_test_static = rng.normal(size=(n_test, d_static))
    y_train = rng.integers(low=0, high=2, size=(n_train,))
    y_test = rng.integers(low=0, high=2, size=(n_test,))

    X_train_ts = {
        "labs": rng.normal(size=(n_train, t, d_ts)),
        "meds": rng.normal(size=(n_train, t, d_ts)),
    }
    X_test_ts = {
        "labs": rng.normal(size=(n_test, t, d_ts)),
        "meds": rng.normal(size=(n_test, t, d_ts)),
    }

    vocab = np.array(["alpha", "beta", "gamma", "delta", "icu", "ward", "discharge"])
    X_train_text = np.array(
        [" ".join(rng.choice(vocab, size=6, replace=True)) for _ in range(n_train)],
        dtype=object,
    )
    X_test_text = np.array(
        [" ".join(rng.choice(vocab, size=6, replace=True)) for _ in range(n_test)],
        dtype=object,
    )

    # Synthetic pre-computed image embeddings (2-D → treated as precomputed by ImageNN).
    X_train_img = None
    X_test_img = None
    if d_img > 0:
        X_train_img = rng.normal(size=(n_train, d_img)).astype(np.float32)
        X_test_img = rng.normal(size=(n_test, d_img)).astype(np.float32)

    dataset = MultimodalDataset(
        y_train=y_train,
        X_train_static=X_train_static,
        X_test_static=X_test_static,
        X_train_ts=X_train_ts,
        X_test_ts=X_test_ts,
        X_train_text=X_train_text,
        X_test_text=X_test_text,
        X_train_img=X_train_img,
        X_test_img=X_test_img,
        y_test=y_test,
    )

    return {
        "dataset": dataset,
        "model": None,
        "text_backend_kwargs": {},
        "image_backend_kwargs": {"image_encoder": "precomputed"},
    }
