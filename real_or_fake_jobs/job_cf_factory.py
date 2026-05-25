"""Factory for the real-or-fake-jobs counterfactual ablation.

Returns the MultimodalDataset and text backend kwargs ready for
run_distance_ablation (from examples/run_distance_ablation.py).

Text handling
-------------
The dataset has three text fields (description, company_profile, requirements).
The MultimodalDataset carries *description* as its primary X_train_text so that
EarlyFusion / Frankenstein / Combined materialise candidates with description
text by default and text-proximity objectives are always evaluated on the same
field.

The other two field arrays are returned in the output dict
(``"X_train_company_profile"``, ``"X_train_requirements"`` etc.) so that
run_cf_ablation.py can build FieldTextNN generators for each combination.

Can be imported directly by run_cf_ablation.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Make cf_lib importable when called from this subdirectory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cf_lib import MultimodalDataset

TEXT_MODEL = "distilbert-base-uncased"
DATA_DIR   = Path(__file__).parent / "data"


def build_job_dataset(
    *,
    data_dir: Optional[str] = None,
    gpu: Optional[int] = None,
    load_bert: bool = True,
    fusion_strategy: str = "intermediate",
) -> Dict[str, Any]:
    """Load the job-posting MultimodalDataset and (optionally) the DistilBERT backend.

    Parameters
    ----------
    data_dir  : path to the folder containing dataset.npz (default: real_or_fake_jobs/data)
    gpu       : GPU index for the bert backend (None → auto-detect)
    load_bert : if True, load the DistilBERT encoder and pass it as bert backend

    Returns
    -------
    dict with keys:
        "dataset"                  : MultimodalDataset
                                     primary text branch = "description"
        "model"                    : None
        "text_backend_kwargs"      : dict with bert_tokenizer / bert_model / bert_device
        "label_classes"            : list of class name strings
        "y_pred"                   : np.ndarray or None (requires evaluate.py first)
        "X_train_description"      : np.ndarray of training description strings
        "X_test_description"       : np.ndarray of test description strings
        "X_train_company_profile"  : np.ndarray of training company-profile strings
        "X_test_company_profile"   : np.ndarray of test company-profile strings
        "X_train_requirements"     : np.ndarray of training requirements strings
        "X_test_requirements"      : np.ndarray of test requirements strings
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if not root.exists():
        raise FileNotFoundError(f"data_dir does not exist: {root}")

    # ---- data arrays ----
    data          = np.load(root / "dataset.npz", allow_pickle=True)
    label_classes = data["label_classes"].tolist()

    X_train_description     = data["X_train_description"]
    X_test_description      = data["X_test_description"]
    X_train_company_profile = data["X_train_company_profile"]
    X_test_company_profile  = data["X_test_company_profile"]
    X_train_requirements    = data["X_train_requirements"]
    X_test_requirements     = data["X_test_requirements"]

    # Primary text = description; the other two fields are returned separately
    # for FieldTextNN generators and multi-field multimodal generators.
    dataset = MultimodalDataset(
        X_train_static = data["X_train_static"].astype(float),
        y_train        = data["y_train"],
        X_test_static  = data["X_test_static"].astype(float),
        X_train_texts  = {
            "description": X_train_description,
            "company_profile": X_train_company_profile,
            "requirements": X_train_requirements,
        },
        X_test_texts   = {
            "description": X_test_description,
            "company_profile": X_test_company_profile,
            "requirements": X_test_requirements,
        },
        primary_text_name = "description",
        y_test         = data["y_test"],
    )

    # ---- predictions ----
    _pred_filename = {
        "intermediate": "y_pred.npy",
        "early":        "y_pred_early_fusion_mlp.npy",
        "late":         "y_pred_late_fusion_tfidf_logreg_nondp.npy",
    }.get(fusion_strategy, "y_pred.npy")
    y_pred_path = root / _pred_filename
    y_pred = np.load(y_pred_path) if y_pred_path.exists() else None

    # ---- text backend ----
    text_backend_kwargs: Dict[str, Any] = {}
    if load_bert:
        from transformers import AutoModel, AutoTokenizer

        if gpu is not None and torch.cuda.is_available():
            device = f"cuda:{gpu}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[factory] Loading text encoder ({TEXT_MODEL}) on {device} …")
        tokenizer  = AutoTokenizer.from_pretrained(TEXT_MODEL)
        bert_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

        text_backend_kwargs = {
            "bert_tokenizer": tokenizer,
            "bert_model":     bert_model,
            "bert_device":    device,
        }

    return {
        "dataset":                  dataset,
        "model":                    None,
        "text_backend_kwargs":      text_backend_kwargs,
        "label_classes":            label_classes,
        "y_pred":                   y_pred,
        # Per-field arrays for FieldTextNN and multi-field generators
        "X_train_description":      X_train_description,
        "X_test_description":       X_test_description,
        "X_train_company_profile":  X_train_company_profile,
        "X_test_company_profile":   X_test_company_profile,
        "X_train_requirements":     X_train_requirements,
        "X_test_requirements":      X_test_requirements,
    }
