"""Factory for the sepsis counterfactual ablation."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cf_lib import MultimodalDataset
from modeling import SepsisModel

DATA_DIR = Path(__file__).parent / "data"
LABEL_CLASSES = ["no_death", "death"]
TS_NAME = "lab_exams_vitals"


def _resolve_device(gpu: Optional[int]) -> str:
    if gpu is not None and torch.cuda.is_available():
        if gpu >= torch.cuda.device_count():
            raise ValueError(
                f"GPU {gpu} requested but only {torch.cuda.device_count()} available."
            )
        return f"cuda:{gpu}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_sepsis_dataset(
    *,
    fold: int,
    data_dir: Optional[str] = None,
    gpu: Optional[int] = None,
    load_model: bool = True,
) -> Dict[str, Any]:
    """Load one sepsis fold as a MultimodalDataset plus its trained model."""
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    fold_dir = root / f"fold_{int(fold)}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"fold directory does not exist: {fold_dir}")

    X_train_static = np.load(fold_dir / "X_train_static.npy").astype(np.float32)
    X_test_static = np.load(fold_dir / "X_test_static.npy").astype(np.float32)
    X_train_ts = np.load(fold_dir / "X_train_ts.npy").astype(np.float32)
    X_test_ts = np.load(fold_dir / "X_test_ts.npy").astype(np.float32)
    y_train = np.load(fold_dir / "y_train.npy").astype(np.int32)
    y_test = np.load(fold_dir / "y_test.npy").astype(np.int32)

    dataset = MultimodalDataset(
        X_train_static=X_train_static,
        y_train=y_train,
        X_test_static=X_test_static,
        X_train_ts={TS_NAME: X_train_ts},
        X_test_ts={TS_NAME: X_test_ts},
        y_test=y_test,
    )

    y_pred_path = fold_dir / "y_pred.npy"
    y_proba_path = fold_dir / "y_proba.npy"
    y_pred = np.load(y_pred_path) if y_pred_path.exists() else None
    y_proba = np.load(y_proba_path) if y_proba_path.exists() else None

    torch_model = None
    device = None
    if load_model:
        model_path = root / f"best_model_{int(fold)}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"best model not found: {model_path}")
        device = _resolve_device(gpu)
        torch_model = SepsisModel()
        torch_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        torch_model.to(device)
        torch_model.eval()

    return {
        "dataset": dataset,
        "model": torch_model,
        "torch_device": device,
        "text_backend_kwargs": {},
        "image_backend_kwargs": {},
        "label_classes": list(LABEL_CLASSES),
        "fold": int(fold),
        "fold_dir": fold_dir,
        "data_dir": root,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "ts_name": TS_NAME,
    }
