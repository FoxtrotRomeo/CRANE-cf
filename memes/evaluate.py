"""
Evaluate the best saved model on the test set and save outputs.

Outputs in data/
----------------
  classification_report.txt  - sklearn text report (precision/recall/F1 per class)
  y_true.npy                 - integer class indices, shape (n_test,)
  y_pred.npy                 - predicted class indices, shape (n_test,)
  y_proba.npy                - softmax probabilities,  shape (n_test, n_classes)

Run
---
    cd memes
    python evaluate.py --gpu 7
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from encoder_features import (
    FineTuneMultimodalClassifier,
    RawMultimodalDataset,
    build_text_tokenizer,
    get_image_transform,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument(
    "--word2vec-path",
    type=str,
    default=None,
    help="Optional override for the word2vec .kv/.bin file used by checkpoints with the word2vec text encoder.",
)
args = parser.parse_args()

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {args.gpu} requested but only {torch.cuda.device_count()} GPU(s) available."
        )
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = "data/dataset.npz"
MODEL_PATH = "data/best_model.pt"
OUT_DIR = "data"
NUM_WORKERS = 2

# ---------------------------------------------------------------------------
# Load checkpoint and data
# ---------------------------------------------------------------------------
ckpt = torch.load(MODEL_PATH, map_location="cpu")
cfg = ckpt["config"]
label_classes = np.array(ckpt["label_classes"])
N_CLASSES = len(label_classes)

data = np.load(DATA_PATH, allow_pickle=True)
X_test_img = data["X_test_img"]
X_test_text = data["X_test_text"].tolist()
y_test = torch.tensor(data["y_test"], dtype=torch.long)

print(f"Test samples: {len(y_test)} | Classes: {label_classes.tolist()}")


def make_test_loader():
    dataset = RawMultimodalDataset(
        images=X_test_img,
        texts=X_test_text,
        labels=y_test,
        text_encoder_name=cfg["text_encoder"],
        text_backend=build_text_tokenizer(
            cfg["text_encoder"],
            word2vec_path=args.word2vec_path or cfg.get("word2vec_path"),
            vocab_tokens=cfg.get("word2vec_vocab_tokens"),
        ),
        image_transform=get_image_transform(cfg["image_encoder"]),
        max_len=cfg.get("max_len", 128),
    )
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


model = FineTuneMultimodalClassifier(
    text_encoder_name=cfg["text_encoder"],
    image_encoder_name=cfg["image_encoder"],
    n_classes=N_CLASSES,
    text_hidden_dim=cfg["text_hidden_dim"],
    img_hidden_dim=cfg["img_hidden_dim"],
    dropout=cfg["dropout"],
    device=DEVICE,
    word2vec_path=args.word2vec_path or cfg.get("word2vec_path"),
    word2vec_vocab_tokens=cfg.get("word2vec_vocab_tokens"),
)
model.load_state_dict(ckpt["state_dict"])
model.to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
all_proba, all_pred, all_true = [], [], []
with torch.no_grad():
    for batch in make_test_loader():
        logits = model(
            batch["input_ids"].to(DEVICE),
            batch["attention_mask"].to(DEVICE),
            batch["image"].to(DEVICE),
        )
        proba = torch.softmax(logits, dim=1).cpu()
        all_proba.append(proba)
        all_pred += proba.argmax(1).tolist()
        all_true += batch["label"].tolist()

y_true = np.array(all_true, dtype=np.int32)
y_pred = np.array(all_pred, dtype=np.int32)
y_proba = torch.cat(all_proba).numpy().astype(np.float32)

# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------
report = classification_report(y_true, y_pred, target_names=label_classes)
print("\nClassification report:")
print(report)

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# ---------------------------------------------------------------------------
# Save arrays
# ---------------------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "y_true.npy"), y_true)
np.save(os.path.join(OUT_DIR, "y_pred.npy"), y_pred)
np.save(os.path.join(OUT_DIR, "y_proba.npy"), y_proba)

print(f"\nSaved to {OUT_DIR}/:")
print("  classification_report.txt")
print(f"  y_true.npy   shape={y_true.shape}")
print(f"  y_pred.npy   shape={y_pred.shape}")
print(f"  y_proba.npy  shape={y_proba.shape}")
