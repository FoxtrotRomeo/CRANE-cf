"""
Early-fusion classifier for Hateful Memes.

Fusion strategy: each modality is first encoded into a fixed-size vector using
a frozen pre-trained encoder; the vectors are concatenated and fed to a
classifier.

  Text branch:  frozen encoder embedding  (768 / 384 / 300-dim)
  Image branch: frozen encoder embedding  (2048 / 1280 / 768-dim)
  Concatenated: [text_emb ‖ img_emb]

NOTE: the existing model (hparam_search.py) is an *intermediate-fusion* model
with learnable projection heads per modality.  This script adds the early-fusion
counterpart.

Encoding (Option A): embeddings are taken from *pre-trained frozen* encoders,
identical to the search-phase cache in hparam_search.py (precomputed_embeddings.pt).
All 3×3 = 9 encoder combinations are swept.

Three classifier families are trained and compared for each combination:

  1. MLP (deep)   — 2-layer feed-forward on the concatenated vector.
  2. RF  (non-deep) — RandomForestClassifier.
  3. GBT (non-deep) — GradientBoostingClassifier.

The combination with the best validation macro-F1 is saved for each family.

Saved as:
  data/best_model_early_fusion_mlp.pt
  data/best_model_early_fusion_rf.pkl
  data/best_model_early_fusion_gbt.pkl

A results CSV with all 9 × 3 = 27 rows is also written to:
  data/early_fusion_results.csv

Run
---
    cd memes
    python train_early_fusion.py --gpu 7
"""

import argparse
import csv
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

from encoder_features import (
    DEFAULT_CACHE_PATH,
    IMAGE_ENCODERS,
    TEXT_ENCODERS,
    load_or_precompute_embeddings,
    collect_word2vec_tokens,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",               type=int,   default=None)
parser.add_argument("--seed",              type=int,   default=42)
parser.add_argument("--clf-epochs",        type=int,   default=50)
parser.add_argument("--clf-lr",            type=float, default=1e-3)
parser.add_argument("--rf-n-estimators",   type=int,   default=300)
parser.add_argument("--gbt-n-estimators",  type=int,   default=200)
parser.add_argument("--gbt-max-depth",     type=int,   default=5)
parser.add_argument("--word2vec-path",     type=str,
                    default="../real_or_fake_jobs/data/word2vec_google_news_300.kv")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = "data/dataset.npz"
OUT_DIR   = "data"
VAL_FRAC  = 0.15
SEED      = args.seed
RESULTS_CSV = os.path.join(OUT_DIR, "early_fusion_results.csv")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if args.gpu is not None and torch.cuda.is_available():
    if args.gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU {args.gpu} not available.")
    DEVICE = f"cuda:{args.gpu}"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH, allow_pickle=True)

X_train_img_all  = data["X_train_img"]
X_test_img       = data["X_test_img"]
X_train_text_all = data["X_train_text"].tolist()
X_test_text      = data["X_test_text"].tolist()
y_train_all      = torch.tensor(data["y_train"], dtype=torch.long)
y_test           = data["y_test"].astype(int)
label_classes    = data["label_classes"].tolist()

N_CLASSES = len(label_classes)
N_TRAIN   = len(y_train_all)
print(f"Classes ({N_CLASSES}): {label_classes}  |  N_train={N_TRAIN}  |  N_test={len(y_test)}")

word2vec_vocab = collect_word2vec_tokens(
    X_train_text_all + X_test_text, args.word2vec_path
)

# Validation split — identical to hparam_search.py
rng_split = random.Random(SEED)
all_idx   = list(range(N_TRAIN))
rng_split.shuffle(all_idx)
n_val         = int(N_TRAIN * VAL_FRAC)
val_indices   = all_idx[:n_val]
train_indices = all_idx[n_val:]
print(f"Split — train: {len(train_indices)}  val: {len(val_indices)}")

y_tr  = y_train_all[train_indices].numpy().astype(int)
y_val = y_train_all[val_indices].numpy().astype(int)

# ---------------------------------------------------------------------------
# Load precomputed embeddings  (same cache as hparam_search.py search stage)
# ---------------------------------------------------------------------------
embedding_cache = load_or_precompute_embeddings(
    X_train_text=X_train_text_all,
    X_test_text=X_test_text,
    X_train_img=X_train_img_all,
    X_test_img=X_test_img,
    device=DEVICE,
    batch_size=64,
    max_len=128,
    num_workers=2,
    cache_path=DEFAULT_CACHE_PATH,
    word2vec_path=args.word2vec_path,
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EarlyFusionMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),     nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Helper: train MLP on a feature matrix, return best val f1 + state
# ---------------------------------------------------------------------------
def train_mlp(X_tr, y_tr, X_val, y_val, d_in, epochs, lr):
    tr_ds  = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                            torch.tensor(y_tr, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))
    tr_dl  = DataLoader(tr_ds,  batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    mlp  = EarlyFusionMLP(d_in, N_CLASSES, hidden=256, dropout=0.3).to(DEVICE)
    opt  = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_vf1, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        mlp.train()
        for xb, yb in tr_dl:
            loss = crit(mlp(xb.to(DEVICE)), yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        mlp.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                pv += mlp(xb.to(DEVICE)).argmax(1).cpu().tolist()
                lv += yb.tolist()
        vf1 = f1_score(lv, pv, average="macro", zero_division=0)
        if vf1 > best_vf1:
            best_vf1   = vf1
            best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
    return best_vf1, best_state, mlp


def eval_mlp(mlp, state, X_test, n_classes):
    mlp.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(X_test, dtype=torch.float32, device=DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy().astype("float32")


# ---------------------------------------------------------------------------
# Sweep all encoder combinations
# ---------------------------------------------------------------------------
csv_rows = []
best_per_family = {
    "mlp": {"vf1": -1.0, "state": None, "d_in": None, "cfg": None, "test_proba": None},
    "rf":  {"vf1": -1.0, "model": None, "cfg": None, "test_proba": None},
    "gbt": {"vf1": -1.0, "model": None, "cfg": None, "test_proba": None},
}

for t_enc in TEXT_ENCODERS:
    for i_enc in IMAGE_ENCODERS:
        print(f"\n{'='*55}")
        print(f"Encoder pair: text={t_enc}  image={i_enc}")
        print(f"{'='*55}")

        t_tr   = embedding_cache["text"][t_enc]["train"][train_indices]
        t_val  = embedding_cache["text"][t_enc]["train"][val_indices]
        t_test = embedding_cache["text"][t_enc]["test"]
        i_tr   = embedding_cache["image"][i_enc]["train"][train_indices]
        i_val  = embedding_cache["image"][i_enc]["train"][val_indices]
        i_test = embedding_cache["image"][i_enc]["test"]

        X_tr_ef   = np.concatenate([t_tr.numpy(),   i_tr.numpy()],   axis=1).astype("float32")
        X_val_ef  = np.concatenate([t_val.numpy(),  i_val.numpy()],  axis=1).astype("float32")
        X_test_ef = np.concatenate([t_test.numpy(), i_test.numpy()], axis=1).astype("float32")
        d_in = X_tr_ef.shape[1]
        print(f"Feature dim: {d_in}")

        # -- MLP
        vf1_mlp, state_mlp, mlp_model = train_mlp(
            X_tr_ef, y_tr, X_val_ef, y_val, d_in,
            epochs=args.clf_epochs, lr=args.clf_lr,
        )
        print(f"  MLP  val macro-F1: {vf1_mlp:.4f}")

        # -- RF
        rf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            class_weight="balanced", n_jobs=-1, random_state=SEED,
        )
        rf.fit(X_tr_ef, y_tr)
        vf1_rf = f1_score(y_val, rf.predict(X_val_ef), average="macro", zero_division=0)
        print(f"  RF   val macro-F1: {vf1_rf:.4f}")

        # -- GBT
        sw  = compute_sample_weight("balanced", y_tr)
        gbt = GradientBoostingClassifier(
            n_estimators=args.gbt_n_estimators,
            max_depth=args.gbt_max_depth,
            learning_rate=0.1, random_state=SEED,
        )
        gbt.fit(X_tr_ef, y_tr, sample_weight=sw)
        vf1_gbt = f1_score(y_val, gbt.predict(X_val_ef), average="macro", zero_division=0)
        print(f"  GBT  val macro-F1: {vf1_gbt:.4f}")

        cfg = {"text_encoder": t_enc, "image_encoder": i_enc, "d_in": d_in}

        # Test probabilities (computed once per combo, reused for best tracking)
        test_proba_mlp = eval_mlp(mlp_model, state_mlp, X_test_ef, N_CLASSES)
        test_proba_rf  = rf.predict_proba(X_test_ef).astype("float32")
        test_proba_gbt = gbt.predict_proba(X_test_ef).astype("float32")

        # Track best per family
        if vf1_mlp > best_per_family["mlp"]["vf1"]:
            best_per_family["mlp"].update(
                vf1=vf1_mlp, state=state_mlp, d_in=d_in,
                cfg=cfg.copy(), test_proba=test_proba_mlp,
            )
        if vf1_rf > best_per_family["rf"]["vf1"]:
            best_per_family["rf"].update(
                vf1=vf1_rf, model=rf, cfg=cfg.copy(), test_proba=test_proba_rf,
            )
        if vf1_gbt > best_per_family["gbt"]["vf1"]:
            best_per_family["gbt"].update(
                vf1=vf1_gbt, model=gbt, cfg=cfg.copy(), test_proba=test_proba_gbt,
            )

        # Log row
        csv_rows.append({
            "text_encoder": t_enc, "image_encoder": i_enc,
            "mlp_val_f1": f"{vf1_mlp:.6f}",
            "rf_val_f1":  f"{vf1_rf:.6f}",
            "gbt_val_f1": f"{vf1_gbt:.6f}",
        })

# ---------------------------------------------------------------------------
# Save best models per family
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

# -- MLP
best_mlp = best_per_family["mlp"]
mlp_save = EarlyFusionMLP(best_mlp["d_in"], N_CLASSES, hidden=256, dropout=0.3)
torch.save(
    {"state_dict": best_mlp["state"], "config": best_mlp["cfg"],
     "label_classes": label_classes},
    os.path.join(OUT_DIR, "best_model_early_fusion_mlp.pt"),
)
print(f"\nBest MLP: {best_mlp['cfg']}  val F1={best_mlp['vf1']:.4f}")

# -- RF
best_rf = best_per_family["rf"]
with open(os.path.join(OUT_DIR, "best_model_early_fusion_rf.pkl"), "wb") as fh:
    pickle.dump({"model": best_rf["model"], "config": best_rf["cfg"],
                 "label_classes": label_classes}, fh)
print(f"Best RF:  {best_rf['cfg']}  val F1={best_rf['vf1']:.4f}")

# -- GBT
best_gbt = best_per_family["gbt"]
with open(os.path.join(OUT_DIR, "best_model_early_fusion_gbt.pkl"), "wb") as fh:
    pickle.dump({"model": best_gbt["model"], "config": best_gbt["cfg"],
                 "label_classes": label_classes}, fh)
print(f"Best GBT: {best_gbt['cfg']}  val F1={best_gbt['vf1']:.4f}")

# ---------------------------------------------------------------------------
# Test evaluation for each best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Test results")
print("=" * 55)
for family, info in best_per_family.items():
    y_pred_t = info["test_proba"].argmax(1)
    print(f"\n[{family.upper()}]  config: {info['cfg']}")
    print(classification_report(y_test, y_pred_t, target_names=label_classes))
    tag = f"early_fusion_{family}"
    np.save(os.path.join(OUT_DIR, f"y_proba_{tag}.npy"), info["test_proba"])
    np.save(os.path.join(OUT_DIR, f"y_pred_{tag}.npy"),  y_pred_t.astype(np.int32))
    np.save(os.path.join(OUT_DIR, f"y_true_{tag}.npy"),  y_test.astype(np.int32))
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as fh:
        fh.write(f"Early-fusion [{family.upper()}]  config: {info['cfg']}\n\n")
        fh.write(classification_report(y_test, y_pred_t, target_names=label_classes))

# ---------------------------------------------------------------------------
# Write results CSV
# ---------------------------------------------------------------------------
with open(RESULTS_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh, fieldnames=["text_encoder", "image_encoder",
                        "mlp_val_f1", "rf_val_f1", "gbt_val_f1"]
    )
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nFull results written to {RESULTS_CSV}")
