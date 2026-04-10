"""
Retrain the SepsisModel (PyTorch) from scratch on each of the 5 cross-validation folds.

Optimises for AUROC on a held-out validation split (15 % of training data).
The test set is never seen during training; it is only used in evaluate.py.

Run
---
    cd ..
    python sepsis/train_pytorch.py --gpu 0

Outputs in sepsis/data/
-----------------------
  best_model_0.pt … best_model_4.pt  — best model per fold (highest test AUROC)
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SpatiotemporalSumAttention(nn.Module):
    """
    Learnable temporal + spatial attention combined by summation.

    Forward pass (x: B × T × F):
        a_time = softmax(W_t @ x,        dim=1)   # (B, T, F)
        a_feat = softmax(W_f @ x^T, dim=1)^T      # (B, T, F)
        out    = (a_time + a_feat) * x
    """

    def __init__(self, n_timesteps: int, n_features: int):
        super().__init__()
        self.w_timesteps = nn.Parameter(torch.empty(n_timesteps, n_timesteps))
        self.w_features  = nn.Parameter(torch.empty(n_features,  n_features))
        nn.init.normal_(self.w_timesteps)
        nn.init.normal_(self.w_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_time = torch.einsum("tj,bjf->btf", self.w_timesteps, x)
        a_time = torch.softmax(a_time, dim=1)
        x_perm = x.permute(0, 2, 1)
        a_feat = torch.einsum("fg,bgT->bfT", self.w_features, x_perm)
        a_feat = torch.softmax(a_feat, dim=1).permute(0, 2, 1)
        return (a_time + a_feat) * x


class SepsisModel(nn.Module):
    """
    Inputs:
        x_ts     : (B, 24, 53)  — time-varying features
        x_static : (B,     47)  — static features (repeated across timesteps)

    Static features are tiled to (B, 24, 47) and concatenated with x_ts
    before the attention + GRU pipeline, giving a (B, 24, 100) tensor.
    """

    def __init__(self, n_timesteps: int = 24,
                 n_ts_features: int = 53, n_static_features: int = 47,
                 gru_units: int = 16, dense1_units: int = 53,
                 dropout_rate: float = 0.2):
        super().__init__()
        n_features        = n_ts_features + n_static_features
        self.n_timesteps  = n_timesteps
        self.attention    = SpatiotemporalSumAttention(n_timesteps, n_features)
        self.gru          = nn.GRU(input_size=n_features, hidden_size=gru_units, batch_first=True)
        self.dropout1     = nn.Dropout(dropout_rate)
        self.dense1       = nn.Linear(gru_units, dense1_units)
        self.bn           = nn.BatchNorm1d(dense1_units, momentum=0.01, eps=0.001)
        self.dropout2     = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(dense1_units, 1)

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_ts, x_static.unsqueeze(1).expand(-1, self.n_timesteps, -1)], dim=2)
        x = self.attention(x)
        _, h = self.gru(x)
        h = self.dropout1(h.squeeze(0))
        h = torch.tanh(self.dense1(h))
        h = self.dropout2(self.bn(h))
        return torch.sigmoid(self.output_layer(h))


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",          type=int,   default=0,
                    help="GPU index to use (e.g. 0). Defaults to CPU if unavailable.")
parser.add_argument("--epochs",       type=int,   default=300,
                    help="Max training epochs per fold (default: 300).")
parser.add_argument("--batch-size",   type=int,   default=64)
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--patience",     type=int,   default=40,
                    help="Early-stopping patience in epochs (default: 40).")
parser.add_argument("--val-frac",     type=float, default=0.15,
                    help="Fraction of training data held out for validation (default: 0.15).")
parser.add_argument("--seed",         type=int,   default=0)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("sepsis/data")
SEED     = args.seed
N_FOLDS  = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only "
                         f"{torch.cuda.device_count()} GPU(s) available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weighted_bce(pred, target, pos_weight):
    eps = 1e-7
    pred = pred.clamp(eps, 1.0 - eps)
    return (-(pos_weight * target * torch.log(pred)
              + (1.0 - target) * torch.log(1.0 - pred))).mean()


def auroc(y_true, y_score):
    order  = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos  = y_true.sum()
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(y_true)     / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y_true) / n_neg])
    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# Training loop — one fold at a time
# ---------------------------------------------------------------------------
fold_results = []

for fold in range(N_FOLDS):
    fold_dir = DATA_DIR / f"fold_{fold}"
    print(f"\n{'='*60}")
    print(f"Fold {fold}/{N_FOLDS - 1}")
    print(f"{'='*60}")

    # Load fold data
    X_ts     = np.load(fold_dir / "X_train_ts.npy").astype("float32")
    X_static = np.load(fold_dir / "X_train_static.npy").astype("float32")
    y_all    = np.load(fold_dir / "y_train.npy").astype("float32")

    # Stratified train / validation split (test set never seen during training)
    idx = np.arange(len(y_all))
    tr_idx, val_idx = train_test_split(idx, test_size=args.val_frac,
                                       stratify=y_all, random_state=SEED)

    X_train_ts,  X_val_ts     = X_ts[tr_idx],     X_ts[val_idx]
    X_train_static, X_val_static = X_static[tr_idx], X_static[val_idx]
    y_train, y_val             = y_all[tr_idx],    y_all[val_idx]

    n_pos      = y_train.sum()
    n_neg      = len(y_train) - n_pos
    pos_weight = n_neg / n_pos
    print(f"Train: {len(X_train_ts)} samples  |  pos={int(n_pos)}, neg={int(n_neg)}  |  pos_weight={pos_weight:.2f}")
    print(f"Val:   {len(X_val_ts)} samples   |  pos={int(y_val.sum())}, neg={int((y_val == 0).sum())}")

    train_ds = TensorDataset(
        torch.tensor(X_train_ts),
        torch.tensor(X_train_static),
        torch.tensor(y_train),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    X_val_ts_t     = torch.tensor(X_val_ts,     device=DEVICE)
    X_val_static_t = torch.tensor(X_val_static, device=DEVICE)

    # Fresh model for each fold
    model = SepsisModel(
        n_timesteps=24, n_ts_features=53, n_static_features=47,
        gru_units=16, dense1_units=53, dropout_rate=0.2,
    ).to(DEVICE)

    p = float(y_train.mean())
    with torch.no_grad():
        model.output_layer.bias.fill_(math.log(p / (1.0 - p)))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-5,
    )

    best_auroc, best_state, patience_counter = 0.0, None, 0

    print(f"\n{'Epoch':>6}  {'Train loss':>11}  {'Val AUROC':>10}  {'LR':>8}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb_ts, xb_static, yb in train_dl:
            xb_ts, xb_static, yb = xb_ts.to(DEVICE), xb_static.to(DEVICE), yb.to(DEVICE)
            pred = model(xb_ts, xb_static).squeeze(1)
            loss = weighted_bce(pred, yb, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(xb_ts)

        avg_loss = total_loss / len(X_train_ts)

        model.eval()
        with torch.no_grad():
            preds_np = model(X_val_ts_t, X_val_static_t).squeeze(1).cpu().numpy()
        val_auc = auroc(y_val, preds_np)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)

        print(f"{epoch:>6}  {avg_loss:>11.4f}  {val_auc:>10.4f}  {current_lr:>8.2e}")

        if val_auc > best_auroc:
            best_auroc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    out_path = DATA_DIR / f"best_model_{fold}.pt"
    torch.save(best_state, out_path)
    fold_results.append((fold, best_auroc))
    print(f"\nFold {fold} best val AUROC: {best_auroc:.4f}  ->  saved to {out_path}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Summary:")
for fold, auc in fold_results:
    print(f"  Fold {fold}: val AUROC {auc:.4f}")
mean_auc = np.mean([auc for _, auc in fold_results])
print(f"  Mean val AUROC: {mean_auc:.4f}")
