"""
Evaluate the best saved model for each fold on its test set and save outputs.

Outputs in sepsis/data/
-----------------------
  fold_N/classification_report.txt  — per-fold sklearn report + AUROC
  fold_N/y_true.npy                 — integer class indices, shape (n_test,)
  fold_N/y_pred.npy                 — predicted class indices (threshold 0.5), shape (n_test,)
  fold_N/y_proba.npy                — positive-class probabilities, shape (n_test,)
  classification_report.txt         — pooled report across all folds

Run
---
    cd ..
    python sepsis/evaluate.py --gpu 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from modeling import SepsisModel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0,
                    help="GPU index to use (e.g. 0). Defaults to CPU if unavailable.")
args = parser.parse_args()

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only "
                         f"{torch.cuda.device_count()} GPU(s) available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR      = Path("sepsis/data")
N_FOLDS       = 5
LABEL_CLASSES = ["no_death", "death"]

# ---------------------------------------------------------------------------
# Per-fold inference
# ---------------------------------------------------------------------------
all_y_true  = []
all_y_pred  = []
all_y_proba = []

for fold in range(N_FOLDS):
    fold_dir = DATA_DIR / f"fold_{fold}"
    print(f"\n{'='*50}")
    print(f"Fold {fold}")
    print(f"{'='*50}")

    X_test_ts     = torch.tensor(np.load(fold_dir / "X_test_ts.npy"),     device=DEVICE)
    X_test_static = torch.tensor(np.load(fold_dir / "X_test_static.npy"), device=DEVICE)
    y_test        = np.load(fold_dir / "y_test.npy").astype(np.int32)

    print(f"Test samples: {len(y_test)}  |  pos={y_test.sum()}, neg={(y_test == 0).sum()}")

    model = SepsisModel(
        n_timesteps=24, n_ts_features=53, n_static_features=47,
        gru_units=16, dense1_units=53, dropout_rate=0.2,
    )
    model.load_state_dict(torch.load(DATA_DIR / f"best_model_{fold}.pt", map_location="cpu"))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        y_proba = model(X_test_ts, X_test_static).squeeze(1).cpu().numpy().astype(np.float32)

    y_pred = (y_proba >= 0.5).astype(np.int32)

    fold_auroc  = roc_auc_score(y_test, y_proba)
    fold_report = classification_report(y_test, y_pred, target_names=LABEL_CLASSES)

    print(f"AUROC: {fold_auroc:.4f}")
    print(fold_report)

    # Save per-fold outputs
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(fold_dir / "classification_report.txt", "w") as f:
        f.write(f"AUROC: {fold_auroc:.4f}\n\n")
        f.write(fold_report)
    np.save(fold_dir / "y_true.npy",  y_test)
    np.save(fold_dir / "y_pred.npy",  y_pred)
    np.save(fold_dir / "y_proba.npy", y_proba)

    all_y_true.append(y_test)
    all_y_pred.append(y_pred)
    all_y_proba.append(y_proba)

# ---------------------------------------------------------------------------
# Pooled report across all folds
# ---------------------------------------------------------------------------
y_true_all  = np.concatenate(all_y_true)
y_pred_all  = np.concatenate(all_y_pred)
y_proba_all = np.concatenate(all_y_proba)

pooled_auroc  = roc_auc_score(y_true_all, y_proba_all)
pooled_report = classification_report(y_true_all, y_pred_all, target_names=LABEL_CLASSES)

print(f"\n{'='*50}")
print("Pooled results across all folds")
print(f"{'='*50}")
print(f"AUROC: {pooled_auroc:.4f}")
print(pooled_report)

with open(DATA_DIR / "classification_report.txt", "w") as f:
    f.write(f"Pooled results across all {N_FOLDS} folds\n")
    f.write(f"AUROC: {pooled_auroc:.4f}\n\n")
    f.write(pooled_report)

np.save(DATA_DIR / "y_true.npy",  y_true_all)
np.save(DATA_DIR / "y_pred.npy",  y_pred_all)
np.save(DATA_DIR / "y_proba.npy", y_proba_all)

print(f"\nSaved per-fold outputs to sepsis/data/fold_N/")
print(f"Saved pooled outputs to sepsis/data/")
