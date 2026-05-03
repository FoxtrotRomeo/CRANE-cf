"""
Late-fusion Sepsis classifier.

Fusion strategy: each modality is processed by a fully independent model up to
the prediction stage; the final probability is the average of the two branches.

  TS branch   →  model_ts(x_ts)     →  p_ts     (scalar in [0,1])
  Static branch →  model_static(x_static) →  p_static (scalar in [0,1])
  Combined    →  (p_ts + p_static) / 2

NOTE: the existing SepsisModel (train_pytorch.py) is an *early-fusion* model.
This script adds the late-fusion counterpart.

Two classifier families are trained per fold:

  Deep options
  ├── TS branch:     GRU(53→32) → Tanh → BN → Dropout → Linear(32→1) → Sigmoid
  └── Static branch: Linear(47→32) → Tanh → BN → Dropout → Linear(32→1) → Sigmoid
  Each branch is trained independently with weighted BCE / Adam / early stopping.
  Saved as:
    best_model_late_fusion_mlp_ts_fold{N}.pt
    best_model_late_fusion_mlp_static_fold{N}.pt

  Non-deep options
  ├── TS branch:     RocketClassifier (aeon) with LogisticRegression(class_weight='balanced')
  └── Static branch: GradientBoostingClassifier (sample_weight balanced)
                     + RandomForestClassifier (class_weight='balanced')  — best selected by val AUROC
  Saved as:
    best_model_late_fusion_rocket_fold{N}.pkl
    best_model_late_fusion_gbt_static_fold{N}.pkl

Evaluation: pooled AUROC, precision/recall/F1 across 5 folds, same protocol as
train_pytorch.py.

aeon input convention: (n_cases, n_channels, n_timepoints).
Sepsis TS arrays are (n, 24, 53) — transposed to (n, 53, 24) before aeon calls.

Run
---
    cd ..
    python sepsis/train_late_fusion.py --gpu 0
"""

import argparse
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from aeon.classification.convolution_based import RocketClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class TSOnlyModel(nn.Module):
    """GRU branch — operates on raw TS features only (no static tiling)."""

    def __init__(
        self,
        n_ts_features: int = 53,
        gru_hidden: int = 32,
        dense_hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru  = nn.GRU(input_size=n_ts_features, hidden_size=gru_hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(gru_hidden, dense_hidden)
        self.bn    = nn.BatchNorm1d(dense_hidden, momentum=0.01, eps=0.001)
        self.out   = nn.Linear(dense_hidden, 1)

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        h = self.drop(h.squeeze(0))
        h = torch.tanh(self.dense(h))
        h = self.bn(h)
        return torch.sigmoid(self.out(h))


class StaticOnlyModel(nn.Module):
    """MLP branch — operates on static features only."""

    def __init__(
        self,
        n_static_features: int = 47,
        hidden: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_static_features, hidden),
            nn.Tanh(),
            nn.BatchNorm1d(hidden, momentum=0.01, eps=0.001),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x_static))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",          type=int,   default=0)
parser.add_argument("--epochs",       type=int,   default=300)
parser.add_argument("--batch-size",   type=int,   default=64)
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.0)
parser.add_argument("--patience",     type=int,   default=40)
parser.add_argument("--val-frac",     type=float, default=0.15)
parser.add_argument("--seed",         type=int,   default=0)
parser.add_argument("--rocket-n-kernels", type=int, default=10_000)
parser.add_argument("--gbt-n-estimators", type=int, default=200)
parser.add_argument("--gbt-max-depth",    type=int, default=4)
parser.add_argument("--rf-n-estimators",  type=int, default=300)
args = parser.parse_args()

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
def weighted_bce(pred, target, pos_weight):
    eps = 1e-7
    pred = pred.clamp(eps, 1.0 - eps)
    return (
        -(pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
    ).mean()


def auroc_np(y_true, y_score):
    order  = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos, n_neg = y_true.sum(), len(y_true) - y_true.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(y_true)     / n_pos])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y_true) / n_neg])
    return float(np.trapz(tpr, fpr))


def train_branch(
    model: nn.Module,
    train_dl: DataLoader,
    val_inputs,          # tuple of tensors already on DEVICE
    y_val: np.ndarray,
    pos_weight: float,
    forward_fn,          # callable: (model, batch) -> pred
    val_forward_fn,      # callable: (model, *val_inputs) -> proba np array
    epochs: int,
    patience: int,
    lr: float,
    wd: float,
    label: str,
) -> dict:
    """Generic training loop for a single branch.  Returns best state dict."""
    p = float(y_val.mean()) if y_val.mean() > 0 else 0.07
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-5
    )
    best_auc, best_state, ctr = 0.0, None, 0

    print(f"\n  Training {label}")
    print(f"  {'Epoch':>6}  {'Loss':>9}  {'Val AUROC':>10}  {'LR':>8}")
    print("  " + "-" * 38)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            pred = forward_fn(model, batch)
            y    = batch[-1].to(DEVICE)
            loss = weighted_bce(pred, y, pos_weight)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(y)
        avg_loss = total_loss / len(train_dl.dataset)

        model.eval()
        with torch.no_grad():
            val_proba = val_forward_fn(model, *val_inputs)
        val_auc = auroc_np(y_val, val_proba)
        scheduler.step(val_auc)
        cur_lr = optimizer.param_groups[0]["lr"]

        if epoch % 20 == 0 or epoch <= 5:
            print(f"  {epoch:>6}  {avg_loss:>9.4f}  {val_auc:>10.4f}  {cur_lr:>8.2e}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ctr = 0
        else:
            ctr += 1
        if ctr >= patience:
            print(f"  Early stopping at epoch {epoch}.")
            break

    print(f"  Best val AUROC [{label}]: {best_auc:.4f}")
    return best_state


@torch.no_grad()
def batch_predict(model, X, device, batch_size=256, forward_fn=None):
    model.eval()
    out = []
    for s in range(0, len(X), batch_size):
        batch = [t[s:s+batch_size].to(device) for t in X]
        out.append(forward_fn(model, *batch).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------
results = {k: [] for k in ("deep", "rocket_gbt", "rocket_rf")}
all_true  = {k: [] for k in results}
all_pred  = {k: [] for k in results}
all_proba = {k: [] for k in results}

for fold in range(N_FOLDS):
    fold_dir = DATA_DIR / f"fold_{fold}"
    print(f"\n{'=' * 60}\nFold {fold}/{N_FOLDS - 1}\n{'=' * 60}")

    # ------------------------------------------------------------------ data
    X_ts     = np.load(fold_dir / "X_train_ts.npy").astype("float32")
    X_static = np.load(fold_dir / "X_train_static.npy").astype("float32")
    y_all    = np.load(fold_dir / "y_train.npy").astype("float32")

    X_test_ts     = np.load(fold_dir / "X_test_ts.npy").astype("float32")
    X_test_static = np.load(fold_dir / "X_test_static.npy").astype("float32")
    y_test        = np.load(fold_dir / "y_test.npy").astype("float32")
    y_test_int    = y_test.astype(int)

    idx = np.arange(len(y_all))
    tr_idx, val_idx = train_test_split(
        idx, test_size=args.val_frac, stratify=y_all, random_state=SEED
    )

    X_tr_ts,  X_val_ts       = X_ts[tr_idx],     X_ts[val_idx]
    X_tr_st,  X_val_st       = X_static[tr_idx], X_static[val_idx]
    y_tr,     y_val           = y_all[tr_idx],    y_all[val_idx]
    y_tr_int                  = y_tr.astype(int)

    pos_w_ts = (y_tr == 0).sum() / (y_tr == 1).sum()
    print(f"Train {len(X_tr_ts)} | pos_weight={pos_w_ts:.2f}")

    # ================================================= DEEP LATE FUSION
    # ---- TS branch
    ts_ds = TensorDataset(torch.tensor(X_tr_ts), torch.tensor(y_tr))
    ts_dl = DataLoader(ts_ds, batch_size=args.batch_size, shuffle=True)

    model_ts = TSOnlyModel().to(DEVICE)
    # bias init
    with torch.no_grad():
        p = float(y_tr.mean())
        model_ts.out.bias.fill_(math.log(p / (1.0 - p)))

    val_ts_t = torch.tensor(X_val_ts, device=DEVICE)
    state_ts = train_branch(
        model_ts, ts_dl,
        val_inputs=(val_ts_t,),
        y_val=y_val,
        pos_weight=pos_w_ts,
        forward_fn=lambda m, b: m(b[0].to(DEVICE)).squeeze(1),
        val_forward_fn=lambda m, x: m(x).squeeze(1).cpu().numpy(),
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, wd=args.weight_decay, label="TS branch (deep)"
    )
    ts_path = DATA_DIR / f"best_model_late_fusion_mlp_ts_fold{fold}.pt"
    torch.save(state_ts, ts_path)

    # ---- Static branch
    st_ds = TensorDataset(torch.tensor(X_tr_st), torch.tensor(y_tr))
    st_dl = DataLoader(st_ds, batch_size=args.batch_size, shuffle=True)

    model_st = StaticOnlyModel().to(DEVICE)
    with torch.no_grad():
        model_st.net[-1].bias.fill_(math.log(p / (1.0 - p)))

    val_st_t = torch.tensor(X_val_st, device=DEVICE)
    state_st = train_branch(
        model_st, st_dl,
        val_inputs=(val_st_t,),
        y_val=y_val,
        pos_weight=pos_w_ts,
        forward_fn=lambda m, b: m(b[0].to(DEVICE)).squeeze(1),
        val_forward_fn=lambda m, x: m(x).squeeze(1).cpu().numpy(),
        epochs=args.epochs, patience=args.patience,
        lr=args.lr, wd=args.weight_decay, label="Static branch (deep)"
    )
    st_path = DATA_DIR / f"best_model_late_fusion_mlp_static_fold{fold}.pt"
    torch.save(state_st, st_path)

    # ---- Combined deep prediction (average of the two branches)
    model_ts.load_state_dict(state_ts)
    model_st.load_state_dict(state_st)
    model_ts.eval()
    model_st.eval()

    with torch.no_grad():
        X_test_ts_t = torch.tensor(X_test_ts, device=DEVICE)
        X_test_st_t = torch.tensor(X_test_static, device=DEVICE)
        p_ts  = model_ts(X_test_ts_t).squeeze(1).cpu().numpy()
        p_st  = model_st(X_test_st_t).squeeze(1).cpu().numpy()

    p_deep = 0.5 * p_ts + 0.5 * p_st
    auc_deep = roc_auc_score(y_test_int, p_deep)
    print(f"\n  Deep late fusion test AUROC: {auc_deep:.4f}")
    print(f"    (saved → {ts_path}, {st_path})")

    # ================================================= NON-DEEP LATE FUSION
    # aeon expects (n_cases, n_channels, n_timepoints) — transpose from (n, T, C)
    X_tr_ts_aeon   = X_tr_ts.transpose(0, 2, 1)    # (n, 53, 24)
    X_test_ts_aeon = X_test_ts.transpose(0, 2, 1)

    # ---- ROCKET for TS branch
    print(f"\n  Training ROCKET (n_kernels={args.rocket_n_kernels}) on TS branch …")
    rocket = RocketClassifier(
        n_kernels=args.rocket_n_kernels,
        estimator=LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED, C=1.0
        ),
        random_state=SEED,
        n_jobs=-1,
    )
    rocket.fit(X_tr_ts_aeon, y_tr_int)
    rocket_path = DATA_DIR / f"best_model_late_fusion_rocket_fold{fold}.pkl"
    with open(rocket_path, "wb") as fh:
        pickle.dump(rocket, fh)

    p_rocket_test = rocket.predict_proba(X_test_ts_aeon)[:, 1].astype(np.float32)
    # Also compute val probabilities for selecting the best static branch below
    p_rocket_val  = rocket.predict_proba(X_val_ts.transpose(0, 2, 1))[:, 1]

    auc_rocket_only = roc_auc_score(y_test_int, p_rocket_test)
    print(f"  ROCKET TS-only test AUROC: {auc_rocket_only:.4f}")

    # ---- Static branch: compare GBT vs RF, pick best by val AUROC
    sw = compute_sample_weight("balanced", y_tr_int)

    print(f"  Training GBT on static features …")
    gbt = GradientBoostingClassifier(
        n_estimators=args.gbt_n_estimators,
        max_depth=args.gbt_max_depth,
        learning_rate=0.1,
        random_state=SEED,
    )
    gbt.fit(X_tr_st, y_tr_int, sample_weight=sw)
    p_gbt_val  = gbt.predict_proba(X_val_st)[:, 1]
    p_gbt_test = gbt.predict_proba(X_test_static)[:, 1].astype(np.float32)

    print(f"  Training RF  on static features …")
    rf = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X_tr_st, y_tr_int)
    p_rf_val  = rf.predict_proba(X_val_st)[:, 1]
    p_rf_test = rf.predict_proba(X_test_static)[:, 1].astype(np.float32)

    # Select best static non-deep model by combined val AUROC with ROCKET
    auc_gbt_combined_val = roc_auc_score(
        y_val.astype(int), 0.5 * p_rocket_val + 0.5 * p_gbt_val
    )
    auc_rf_combined_val  = roc_auc_score(
        y_val.astype(int), 0.5 * p_rocket_val + 0.5 * p_rf_val
    )
    print(f"  Val AUROC  ROCKET+GBT={auc_gbt_combined_val:.4f}  ROCKET+RF={auc_rf_combined_val:.4f}")

    if auc_gbt_combined_val >= auc_rf_combined_val:
        best_static_model = gbt
        p_static_test     = p_gbt_test
        static_label      = "gbt"
    else:
        best_static_model = rf
        p_static_test     = p_rf_test
        static_label      = "rf"

    gbt_path = DATA_DIR / f"best_model_late_fusion_gbt_static_fold{fold}.pkl"
    with open(gbt_path, "wb") as fh:
        pickle.dump(best_static_model, fh)

    # ---- Combined non-deep prediction
    p_nondp = 0.5 * p_rocket_test + 0.5 * p_static_test
    auc_nondp = roc_auc_score(y_test_int, p_nondp)
    y_pred_nondp = (p_nondp >= 0.5).astype(np.int32)
    print(f"  Non-deep late fusion ({static_label} static) test AUROC: {auc_nondp:.4f}")

    # ---- Also save pure ROCKET+GBT combination regardless (for reference)
    p_rocket_gbt = 0.5 * p_rocket_test + 0.5 * p_gbt_test
    auc_rocket_gbt = roc_auc_score(y_test_int, p_rocket_gbt)

    # ------------------------------------------------- accumulate
    y_pred_deep  = (p_deep  >= 0.5).astype(np.int32)
    y_pred_rktgbt = (p_rocket_gbt >= 0.5).astype(np.int32)

    results["deep"].append(auc_deep)
    results["rocket_gbt"].append(auc_rocket_gbt)
    results["rocket_rf"].append(roc_auc_score(y_test_int, 0.5*p_rocket_test+0.5*p_rf_test))

    for key, yp, ypr in [
        ("deep",       y_pred_deep,   p_deep),
        ("rocket_gbt", y_pred_rktgbt, p_rocket_gbt),
        ("rocket_rf",  (0.5*p_rocket_test+0.5*p_rf_test >= 0.5).astype(np.int32),
                       0.5*p_rocket_test+0.5*p_rf_test),
    ]:
        all_true[key].append(y_test_int)
        all_pred[key].append(yp)
        all_proba[key].append(ypr.astype(np.float32))

# ---------------------------------------------------------------------------
# Pooled summary
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("Late-fusion results — pooled across 5 folds")
print("=" * 60)

labels = {
    "deep":       "Deep  (GRU-TS + MLP-static, averaged)",
    "rocket_gbt": "ROCKET-TS + GBT-static, averaged",
    "rocket_rf":  "ROCKET-TS + RF-static,  averaged",
}

for key, label in labels.items():
    y_true_all  = np.concatenate(all_true[key])
    y_pred_all  = np.concatenate(all_pred[key])
    y_proba_all = np.concatenate(all_proba[key])

    pooled_auroc  = roc_auc_score(y_true_all, y_proba_all)
    pooled_report = classification_report(
        y_true_all, y_pred_all, target_names=LABEL_CLASSES
    )
    per_fold = "  ".join(f"F{i}={a:.3f}" for i, a in enumerate(results[key]))

    print(f"\n[{label}]")
    print(f"  per-fold: {per_fold}")
    print(f"  pooled AUROC: {pooled_auroc:.4f}")
    print(pooled_report)

    tag = f"late_fusion_{key}"
    np.save(DATA_DIR / f"y_true_{tag}.npy",  y_true_all)
    np.save(DATA_DIR / f"y_pred_{tag}.npy",  y_pred_all)
    np.save(DATA_DIR / f"y_proba_{tag}.npy", y_proba_all)
    with open(DATA_DIR / f"classification_report_{tag}.txt", "w") as f:
        f.write(f"Late-fusion [{label}] — pooled across {N_FOLDS} folds\n")
        f.write(f"Per-fold AUROC: {per_fold}\n")
        f.write(f"Pooled AUROC: {pooled_auroc:.4f}\n\n")
        f.write(pooled_report)
