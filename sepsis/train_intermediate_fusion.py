"""
Intermediate-fusion Sepsis classifier.

Fusion strategy: each modality is encoded by its own branch before being combined.

  TS branch:     GRU(input=53, hidden=32) → last hidden state          (B, 32)
  Static branch: Linear(47→32) → ReLU → Dropout                        (B, 32)
  Fusion:        concat(64) → Linear(64→32) → Tanh → BN → Dropout
                 → Linear(32→1) → Sigmoid

NOTE: the existing SepsisModel (train_pytorch.py) is an *early-fusion* model —
it tiles the static features across timesteps and concatenates them with the TS
at the raw-feature level before any encoder runs.  This script introduces the
intermediate-fusion counterpart, where each modality gets its own encoder first.

Two classifier families are trained and compared per fold:

  1. MLP (deep)   — end-to-end neural IF model trained with the same
                    weighted-BCE / Adam / early-stopping protocol.
                    Saved as: best_model_intermediate_fusion_mlp_fold{N}.pt

  2. RF  (non-deep) — the 32-dim penultimate latent vectors from the
  3. GBT (non-deep)   trained deep IF model are extracted and used as features
                    for a Random Forest and a Gradient Boosting classifier.
                    Saved as: best_model_intermediate_fusion_rf_fold{N}.pkl
                              best_model_intermediate_fusion_gbt_fold{N}.pkl

All three classifiers are evaluated on the held-out test set for each fold;
pooled AUROC, precision, recall and F1 are reported exactly as in evaluate.py.

Run
---
    cd ..
    python sepsis/train_intermediate_fusion.py --gpu 0
"""

import argparse
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SepsisIFModel(nn.Module):
    """
    Intermediate-fusion model: separate GRU encoder for the TS branch and a
    small MLP encoder for the static branch.  The two latent vectors are
    concatenated and passed through a fusion MLP before the output layer.

    Inputs
    ------
    x_ts     : (B, 24, 53)  — time-varying features (no static concatenated)
    x_static : (B,     47)  — static features

    The `encode` method returns the 32-dim fused latent vector that is used as
    input to the non-deep (RF / GBT) classifiers.
    """

    def __init__(
        self,
        n_ts_features: int = 53,
        n_static_features: int = 47,
        ts_hidden: int = 32,
        static_hidden: int = 32,
        fused_hidden: int = 32,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_ts_features, hidden_size=ts_hidden, batch_first=True
        )
        self.ts_drop = nn.Dropout(dropout_rate)

        self.static_enc = nn.Sequential(
            nn.Linear(n_static_features, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        fusion_in = ts_hidden + static_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fused_hidden),
            nn.Tanh(),
            nn.BatchNorm1d(fused_hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Linear(fused_hidden, 1)

    def encode(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """Return the 32-dim fused latent vector (before output layer)."""
        _, h = self.gru(x_ts)
        ts_emb = self.ts_drop(h.squeeze(0))          # (B, ts_hidden)
        static_emb = self.static_enc(x_static)        # (B, static_hidden)
        return self.fusion(torch.cat([ts_emb, static_emb], dim=1))  # (B, fused_hidden)

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output_layer(self.encode(x_ts, x_static)))


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--patience", type=int, default=40)
parser.add_argument("--val-frac", type=float, default=0.15)
parser.add_argument("--seed", type=int, default=0)
# GBT / RF options
parser.add_argument("--rf-n-estimators",  type=int, default=300)
parser.add_argument("--gbt-n-estimators", type=int, default=200)
parser.add_argument("--gbt-max-depth",    type=int, default=4)
parser.add_argument("--gbt-lr",           type=float, default=0.1)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("sepsis/data")
SEED     = args.seed
N_FOLDS  = 5
LABEL_CLASSES = ["no_death", "death"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {args.gpu} requested but only {torch.cuda.device_count()} available."
        )
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    eps = 1e-7
    pred = pred.clamp(eps, 1.0 - eps)
    return (
        -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    ).mean()


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order  = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos  = y_true.sum()
    n_neg  = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(y_true)     / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y_true) / n_neg])
    return float(np.trapz(tpr, fpr))


@torch.no_grad()
def extract_latents(
    model: SepsisIFModel,
    X_ts: np.ndarray,
    X_static: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract fused latent representations from the IF model (no gradient)."""
    model.eval()
    latents = []
    n = len(X_ts)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        x_ts = torch.tensor(X_ts[start:end], dtype=torch.float32, device=device)
        x_st = torch.tensor(X_static[start:end], dtype=torch.float32, device=device)
        latents.append(model.encode(x_ts, x_st).cpu().numpy())
    return np.concatenate(latents, axis=0)


@torch.no_grad()
def predict_proba_deep(
    model: SepsisIFModel,
    X_ts: np.ndarray,
    X_static: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    probas = []
    n = len(X_ts)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        x_ts = torch.tensor(X_ts[start:end], dtype=torch.float32, device=device)
        x_st = torch.tensor(X_static[start:end], dtype=torch.float32, device=device)
        probas.append(model(x_ts, x_st).squeeze(1).cpu().numpy())
    return np.concatenate(probas).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-fold training loop
# ---------------------------------------------------------------------------
fold_results = {k: [] for k in ("mlp", "rf", "gbt")}

all_y_true  = {k: [] for k in ("mlp", "rf", "gbt")}
all_y_pred  = {k: [] for k in ("mlp", "rf", "gbt")}
all_y_proba = {k: [] for k in ("mlp", "rf", "gbt")}

for fold in range(N_FOLDS):
    fold_dir = DATA_DIR / f"fold_{fold}"
    sep = "=" * 60
    print(f"\n{sep}\nFold {fold}/{N_FOLDS - 1}\n{sep}")

    # ------------------------------------------------------------------ data
    X_ts     = np.load(fold_dir / "X_train_ts.npy").astype("float32")
    X_static = np.load(fold_dir / "X_train_static.npy").astype("float32")
    y_all    = np.load(fold_dir / "y_train.npy").astype("float32")

    X_test_ts     = np.load(fold_dir / "X_test_ts.npy").astype("float32")
    X_test_static = np.load(fold_dir / "X_test_static.npy").astype("float32")
    y_test        = np.load(fold_dir / "y_test.npy").astype("float32")

    idx = np.arange(len(y_all))
    tr_idx, val_idx = train_test_split(
        idx, test_size=args.val_frac, stratify=y_all, random_state=SEED
    )

    X_tr_ts, X_val_ts         = X_ts[tr_idx],     X_ts[val_idx]
    X_tr_static, X_val_static = X_static[tr_idx], X_static[val_idx]
    y_tr, y_val               = y_all[tr_idx],    y_all[val_idx]

    n_pos      = y_tr.sum()
    n_neg      = len(y_tr) - n_pos
    pos_weight = n_neg / n_pos
    print(f"Train {len(X_tr_ts)} | pos={int(n_pos)} neg={int(n_neg)} pos_weight={pos_weight:.2f}")
    print(f"Val   {len(X_val_ts)} | pos={int(y_val.sum())}")

    # -------------------------------------------------------- deep IF training
    train_ds = TensorDataset(
        torch.tensor(X_tr_ts),
        torch.tensor(X_tr_static),
        torch.tensor(y_tr),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    X_val_ts_t     = torch.tensor(X_val_ts,     device=DEVICE)
    X_val_static_t = torch.tensor(X_val_static, device=DEVICE)

    model = SepsisIFModel(
        n_ts_features=53, n_static_features=47,
        ts_hidden=32, static_hidden=32, fused_hidden=32, dropout_rate=0.2,
    ).to(DEVICE)

    p = float(y_tr.mean())
    with torch.no_grad():
        model.output_layer.bias.fill_(math.log(p / (1.0 - p)))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-5
    )

    best_auroc_mlp, best_state, patience_ctr = 0.0, None, 0

    print(f"\n{'Epoch':>6}  {'Train loss':>11}  {'Val AUROC':>10}  {'LR':>8}")
    print("-" * 42)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb_ts, xb_static, yb in train_dl:
            xb_ts, xb_static, yb = (
                xb_ts.to(DEVICE), xb_static.to(DEVICE), yb.to(DEVICE)
            )
            pred = model(xb_ts, xb_static).squeeze(1)
            loss = weighted_bce(pred, yb, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(xb_ts)

        avg_loss = total_loss / len(X_tr_ts)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_ts_t, X_val_static_t).squeeze(1).cpu().numpy()
        val_auc = auroc(y_val, val_preds)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_auc)

        if epoch % 10 == 0 or epoch <= 5:
            print(f"{epoch:>6}  {avg_loss:>11.4f}  {val_auc:>10.4f}  {current_lr:>8.2e}")

        if val_auc > best_auroc_mlp:
            best_auroc_mlp = val_auc
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr   = 0
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    # Save deep IF model
    mlp_path = DATA_DIR / f"best_model_intermediate_fusion_mlp_fold{fold}.pt"
    torch.save(best_state, mlp_path)
    print(f"Deep IF model saved → {mlp_path}  (val AUROC {best_auroc_mlp:.4f})")

    # Evaluate MLP on test set
    model.load_state_dict(best_state)
    y_proba_mlp = predict_proba_deep(model, X_test_ts, X_test_static, DEVICE)
    y_pred_mlp  = (y_proba_mlp >= 0.5).astype(np.int32)
    y_test_int  = y_test.astype(np.int32)
    test_auroc_mlp = roc_auc_score(y_test_int, y_proba_mlp)
    print(f"  MLP test AUROC: {test_auroc_mlp:.4f}")

    # ------------------------------------------------ non-deep IF preparation
    # Extract latent representations from the trained deep IF model.
    # Training-set latents are used to fit RF / GBT; test-set latents for eval.
    # This is the correct intermediate-fusion non-deep approach: the same learned
    # per-modality encoders produce the feature space, only the final classifier
    # is replaced with a tree-based model.

    model.load_state_dict(best_state)
    # Use the FULL training set (tr_idx only, not val) to avoid leakage
    Z_tr   = extract_latents(model, X_tr_ts,     X_tr_static,     DEVICE)
    Z_test = extract_latents(model, X_test_ts,    X_test_static,   DEVICE)

    y_tr_int   = y_tr.astype(int)
    sw_tr      = compute_sample_weight("balanced", y_tr_int)

    # -- Random Forest
    print(f"\nTraining RF (n_estimators={args.rf_n_estimators}) on latents …")
    rf = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(Z_tr, y_tr_int)
    rf_path = DATA_DIR / f"best_model_intermediate_fusion_rf_fold{fold}.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)

    y_proba_rf = rf.predict_proba(Z_test)[:, 1].astype(np.float32)
    y_pred_rf  = rf.predict(Z_test).astype(np.int32)
    test_auroc_rf = roc_auc_score(y_test_int, y_proba_rf)
    print(f"  RF  test AUROC: {test_auroc_rf:.4f}  (saved → {rf_path})")

    # -- Gradient Boosting
    print(f"Training GBT (n_estimators={args.gbt_n_estimators}, depth={args.gbt_max_depth}) on latents …")
    gbt = GradientBoostingClassifier(
        n_estimators=args.gbt_n_estimators,
        max_depth=args.gbt_max_depth,
        learning_rate=args.gbt_lr,
        random_state=SEED,
    )
    gbt.fit(Z_tr, y_tr_int, sample_weight=sw_tr)
    gbt_path = DATA_DIR / f"best_model_intermediate_fusion_gbt_fold{fold}.pkl"
    with open(gbt_path, "wb") as f:
        pickle.dump(gbt, f)

    y_proba_gbt = gbt.predict_proba(Z_test)[:, 1].astype(np.float32)
    y_pred_gbt  = gbt.predict(Z_test).astype(np.int32)
    test_auroc_gbt = roc_auc_score(y_test_int, y_proba_gbt)
    print(f"  GBT test AUROC: {test_auroc_gbt:.4f}  (saved → {gbt_path})")

    # ------------------------------------------------- accumulate fold results
    fold_results["mlp"].append(test_auroc_mlp)
    fold_results["rf"].append(test_auroc_rf)
    fold_results["gbt"].append(test_auroc_gbt)

    for key, yp, ypr in [
        ("mlp", y_pred_mlp, y_proba_mlp),
        ("rf",  y_pred_rf,  y_proba_rf),
        ("gbt", y_pred_gbt, y_proba_gbt),
    ]:
        all_y_true[key].append(y_test_int)
        all_y_pred[key].append(yp)
        all_y_proba[key].append(ypr)

# ---------------------------------------------------------------------------
# Pooled summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Intermediate-fusion results — pooled across 5 folds")
print("=" * 60)

for key in ("mlp", "rf", "gbt"):
    y_true_all  = np.concatenate(all_y_true[key])
    y_pred_all  = np.concatenate(all_y_pred[key])
    y_proba_all = np.concatenate(all_y_proba[key])

    pooled_auroc  = roc_auc_score(y_true_all, y_proba_all)
    pooled_report = classification_report(
        y_true_all, y_pred_all, target_names=LABEL_CLASSES
    )
    per_fold_str = "  ".join(f"F{i}={a:.3f}" for i, a in enumerate(fold_results[key]))

    print(f"\n[{key.upper()}]  per-fold: {per_fold_str}")
    print(f"        pooled AUROC: {pooled_auroc:.4f}")
    print(pooled_report)

    # Save pooled arrays following the same convention as evaluate.py
    tag = f"intermediate_fusion_{key}"
    np.save(DATA_DIR / f"y_true_{tag}.npy",  y_true_all)
    np.save(DATA_DIR / f"y_pred_{tag}.npy",  y_pred_all)
    np.save(DATA_DIR / f"y_proba_{tag}.npy", y_proba_all)
    with open(DATA_DIR / f"classification_report_{tag}.txt", "w") as f:
        f.write(f"Intermediate-fusion [{key.upper()}] — pooled across {N_FOLDS} folds\n")
        f.write(f"Per-fold AUROC: {per_fold_str}\n")
        f.write(f"Pooled AUROC: {pooled_auroc:.4f}\n\n")
        f.write(pooled_report)
