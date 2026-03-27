"""Factory for the Hateful Memes counterfactual ablation.

Returns the MultimodalDataset and backend kwargs ready for
run_distance_ablation (from examples/run_distance_ablation.py).

Can be used via the CLI runner:

    cd cf-lib
    python examples/run_distance_ablation.py \\
        --factory memes.hateful_memes_cf_factory:build_hateful_memes_dataset \\
        --target-value 0 \\
        --output-dir memes/data/ablation_runs \\
        --text-encoders bert,tfidf,raw \\
        --image-encoders resnet50,efficientnet_b0,vit_b_16 \\
        --save-full

Or imported directly by run_cf_ablation.py.
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

TEXT_MODEL = "bert-base-uncased"
DATA_DIR   = Path(__file__).parent / "data"


def build_hateful_memes_dataset(
    *,
    data_dir: Optional[str] = None,
    gpu: Optional[int] = None,
    load_bert: bool = True,
) -> Dict[str, Any]:
    """Load the Hateful Memes MultimodalDataset and (optionally) the BERT backend.

    Parameters
    ----------
    data_dir  : path to the folder containing dataset.npz (default: memes/data)
    gpu       : GPU index for the bert backend (None → CPU)
    load_bert : if True, load DistilBERT and pass it as bert backend

    Returns
    -------
    dict with keys:
        "dataset"              : MultimodalDataset  (text + image branches)
        "model"                : None
        "text_backend_kwargs"  : dict with bert_tokenizer / bert_model / bert_device
        "image_backend_kwargs" : empty dict — encoder chosen per ablation combo
        "label_classes"        : list of class name strings (index = integer label)
        "y_pred"               : np.ndarray of predicted class indices on the test set
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if not root.exists():
        raise FileNotFoundError(f"data_dir does not exist: {root}")

    # ---- data arrays ----
    data          = np.load(root / "dataset.npz", allow_pickle=True)
    label_classes = data["label_classes"].tolist()

    # Text + image only: the current dataset stores cleaned meme text and raw
    # 224x224 RGB images, with no static/tabular modality.
    dataset = MultimodalDataset(
        y_train           = data["y_train"],
        X_train_texts     = {"meme_text": data["X_train_text"]},
        X_test_texts      = {"meme_text": data["X_test_text"]},
        primary_text_name = "meme_text",
        X_train_images    = {"meme_img": data["X_train_img"]},
        X_test_images     = {"meme_img": data["X_test_img"]},
        primary_image_name = "meme_img",
        y_test            = data["y_test"],
    )

    # ---- predictions ----
    y_pred_path = root / "y_pred.npy"
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

    # ---- image backend ----
    # Images are raw pixels (n, 224, 224, 3) uint8 — the encoder is chosen
    # per ablation combo by the caller (e.g. resnet50, efficientnet_b0, vit_b_16).
    image_backend_kwargs: Dict[str, Any] = {}

    return {
        "dataset":              dataset,
        "model":                None,
        "text_backend_kwargs":  text_backend_kwargs,
        "image_backend_kwargs": image_backend_kwargs,
        "label_classes":        label_classes,
        "y_pred":               y_pred,
    }
