"""Factory for the Long COVID tweet counterfactual ablation.

Returns the MultimodalDataset and text backend kwargs ready for
run_distance_ablation (from examples/run_distance_ablation.py).

Can be used via the CLI runner:

    cd cf-lib
    python examples/run_distance_ablation.py \
        --factory long_covid_tweets.tweet_cf_factory:build_tweet_dataset \
        --target-value <joy_idx> \
        --sample-indices <comma-separated sadness test indices> \
        --output-dir long_covid_tweets/data/ablation_runs \
        --text-encoders bert,tfidf,raw \
        --tab-metrics euclidean,manhattan \
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

TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"
DATA_DIR   = Path(__file__).parent / "data"


def build_tweet_dataset(
    *,
    data_dir: Optional[str] = None,
    gpu: Optional[int] = None,
    load_bert: bool = True,
) -> Dict[str, Any]:
    """Load the tweet MultimodalDataset and (optionally) the XLM-RoBERTa text backend.

    Parameters
    ----------
    data_dir  : path to the folder containing dataset.npz (default: long_covid_tweets/data)
    gpu       : GPU index for the bert backend (None → CPU)
    load_bert : if True, load the cached XLM-RoBERTa encoder and pass it as bert backend

    Returns
    -------
    dict with keys:
        "dataset"             : MultimodalDataset
        "model"               : None  (IntermediateFusionNN not used here)
        "text_backend_kwargs" : dict with bert_tokenizer / bert_model / bert_device
        "label_classes"       : list of class name strings (index = integer label)
        "y_pred"              : np.ndarray of predicted class indices on the test set
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    if not root.exists():
        raise FileNotFoundError(f"data_dir does not exist: {root}")

    # ---- data arrays ----
    data          = np.load(root / "dataset.npz", allow_pickle=True)
    label_classes = data["label_classes"].tolist()

    x_train_text = data["X_train_text"]
    x_test_text = data["X_test_text"]

    dataset = MultimodalDataset(
        X_train_static = data["X_train_static"].astype(float),
        y_train        = data["y_train"],
        X_test_static  = data["X_test_static"].astype(float),
        X_train_texts  = {"tweet": x_train_text},
        X_test_texts   = {"tweet": x_test_text},
        primary_text_name = "tweet",
        y_test         = data["y_test"],
    )

    # ---- predictions (for caller to derive sample_indices / target_value) ----
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
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        bert_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()

        text_backend_kwargs = {
            "bert_tokenizer": tokenizer,
            "bert_model":     bert_model,
            "bert_device":    device,
        }

    return {
        "dataset":             dataset,
        "model":               None,
        "text_backend_kwargs": text_backend_kwargs,
        "label_classes":       label_classes,
        "y_pred":              y_pred,
    }
