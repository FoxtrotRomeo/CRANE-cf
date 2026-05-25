"""Generate per-fold y_pred / y_proba for intermediate-RF and late-deep fusion models.

Outputs written to sepsis/data/fold_N/:
  y_pred_intermediate_rf.npy   — RF predictions on test latents
  y_proba_intermediate_rf.npy  — RF positive-class probability
  y_pred_late_deep.npy         — averaged MLP-TS + MLP-static predictions (threshold 0.5)
  y_proba_late_deep.npy        — averaged probabilities before thresholding

Run from repo root:
    python sepsis/evaluate_fusion.py --gpu 0
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Inline model definitions (mirrors run_cf_ablation.py to avoid argparse import)
# ---------------------------------------------------------------------------

class _SepsisIFModel(nn.Module):
    def __init__(self, n_ts=53, n_st=47, ts_h=32, st_h=32, fused_h=32, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_ts, hidden_size=ts_h, batch_first=True)
        self.ts_drop = nn.Dropout(drop)
        self.static_enc = nn.Sequential(nn.Linear(n_st, st_h), nn.ReLU(), nn.Dropout(drop))
        self.fusion = nn.Sequential(
            nn.Linear(ts_h + st_h, fused_h), nn.Tanh(),
            nn.BatchNorm1d(fused_h, momentum=0.01, eps=0.001), nn.Dropout(drop),
        )
        self.output_layer = nn.Linear(fused_h, 1)

    def encode(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        return self.fusion(torch.cat([self.ts_drop(h.squeeze(0)), self.static_enc(x_static)], dim=1))

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.output_layer(self.encode(x_ts, x_static)))


class _TSOnlyModel(nn.Module):
    def __init__(self, n_ts=53, gru_h=32, dense_h=32, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=n_ts, hidden_size=gru_h, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.dense = nn.Linear(gru_h, dense_h)
        self.bn = nn.BatchNorm1d(dense_h, momentum=0.01, eps=0.001)
        self.out = nn.Linear(dense_h, 1)

    def forward(self, x_ts: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x_ts)
        h = torch.tanh(self.dense(self.drop(h.squeeze(0))))
        return torch.sigmoid(self.out(self.bn(h)))


class _StaticOnlyModel(nn.Module):
    def __init__(self, n_st=47, hidden=32, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_st, hidden), nn.Tanh(),
            nn.BatchNorm1d(hidden, momentum=0.01, eps=0.001), nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x_static))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--folds", type=str, default="0,1,2,3,4")
args = parser.parse_args()

folds = [int(f.strip()) for f in args.folds.split(",")]

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} requested but only {torch.cuda.device_count()} available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

DATA_DIR = Path(__file__).resolve().parent / "data"
LABEL_CLASSES = ["no_death", "death"]
BS = 128


def _batch_encode_if(enc, X_ts, X_static, device, bs=BS):
    rows = []
    for s in range(0, len(X_static), bs):
        e = min(s + bs, len(X_static))
        with torch.no_grad():
            lat = enc.encode(
                torch.tensor(X_ts[s:e], dtype=torch.float32, device=device),
                torch.tensor(X_static[s:e], dtype=torch.float32, device=device),
            ).cpu().numpy()
        rows.append(lat)
    return np.vstack(rows).astype(np.float32)


def _batch_infer(model, X_tensor_list, device, bs=BS):
    """Run model on a list of tensors (one per input), return squeezed proba array."""
    rows = []
    n = X_tensor_list[0].shape[0]
    for s in range(0, n, bs):
        e = min(s + bs, n)
        inputs = [t[s:e].to(device) for t in X_tensor_list]
        with torch.no_grad():
            out = model(*inputs).squeeze(1).cpu().numpy()
        rows.append(out)
    return np.concatenate(rows).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

for fold in folds:
    fold_dir = DATA_DIR / f"fold_{fold}"
    print(f"\n{'='*55}\nFold {fold}\n{'='*55}")

    X_test_ts = np.load(fold_dir / "X_test_ts.npy").astype(np.float32)
    X_test_static = np.load(fold_dir / "X_test_static.npy").astype(np.float32)
    y_test = np.load(fold_dir / "y_test.npy").astype(np.int32)

    # ------------------------------------------------------------------
    # Intermediate fusion — RF on latents
    # ------------------------------------------------------------------
    enc = _SepsisIFModel().to(DEVICE)
    enc.load_state_dict(
        torch.load(DATA_DIR / f"best_model_intermediate_fusion_mlp_fold{fold}.pt", map_location="cpu")
    )
    enc.eval()

    with open(DATA_DIR / f"best_model_intermediate_fusion_rf_fold{fold}.pkl", "rb") as fh:
        clf_rf = pickle.load(fh)

    te_lat = _batch_encode_if(enc, X_test_ts, X_test_static, DEVICE)
    y_pred_if = clf_rf.predict(te_lat).astype(np.int32)
    if hasattr(clf_rf, "predict_proba"):
        y_proba_if = clf_rf.predict_proba(te_lat)[:, 1].astype(np.float32)
    else:
        y_proba_if = y_pred_if.astype(np.float32)

    np.save(fold_dir / "y_pred_intermediate_rf.npy", y_pred_if)
    np.save(fold_dir / "y_proba_intermediate_rf.npy", y_proba_if)

    auroc_if = roc_auc_score(y_test, y_proba_if)
    print(f"[Intermediate RF] AUROC={auroc_if:.4f}  pos_predicted={y_pred_if.sum()}")
    print(classification_report(y_test, y_pred_if, target_names=LABEL_CLASSES))

    # ------------------------------------------------------------------
    # Late fusion — averaged MLP-TS + MLP-static
    # ------------------------------------------------------------------
    mts = _TSOnlyModel().to(DEVICE)
    mts.load_state_dict(
        torch.load(DATA_DIR / f"best_model_late_fusion_mlp_ts_fold{fold}.pt", map_location="cpu")
    )
    mts.eval()

    mst = _StaticOnlyModel().to(DEVICE)
    mst.load_state_dict(
        torch.load(DATA_DIR / f"best_model_late_fusion_mlp_static_fold{fold}.pt", map_location="cpu")
    )
    mst.eval()

    ts_tensor = torch.tensor(X_test_ts, dtype=torch.float32)
    st_tensor = torch.tensor(X_test_static, dtype=torch.float32)

    p_ts = _batch_infer(mts, [ts_tensor], DEVICE)
    p_st = _batch_infer(mst, [st_tensor], DEVICE)
    y_proba_late = ((p_ts + p_st) / 2.0).astype(np.float32)
    y_pred_late = (y_proba_late >= 0.5).astype(np.int32)

    np.save(fold_dir / "y_pred_late_deep.npy", y_pred_late)
    np.save(fold_dir / "y_proba_late_deep.npy", y_proba_late)

    auroc_late = roc_auc_score(y_test, y_proba_late)
    print(f"[Late deep]       AUROC={auroc_late:.4f}  pos_predicted={y_pred_late.sum()}")
    print(classification_report(y_test, y_pred_late, target_names=LABEL_CLASSES))

print("\nDone. Per-fold y_pred/y_proba files written to sepsis/data/fold_N/")
