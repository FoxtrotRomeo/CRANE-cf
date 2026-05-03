"""
Early-fusion classifier for Long COVID Tweets.

Fusion strategy: each modality is first encoded into a fixed-size vector using
a frozen pre-trained encoder; the vectors are concatenated into a single feature
vector that is fed directly to the classifier.

  Text branch:     frozen XLM-RoBERTa CLS embedding            (768-dim)
  Tabular branch:  raw scaled tabular features                  (D_TAB-dim)
  Concatenated:    [text_emb ‖ tab_feats]

NOTE: the existing MultimodalClassifier (hparam_search.py) is an
*intermediate-fusion* model — both branches are encoded, projected, then fused
before the classifier.  This script adds the early-fusion counterpart.

Encoding note (Option A): embeddings are extracted from the frozen *pre-trained*
XLM-RoBERTa weights, not from the fine-tuned classifier checkpoint.  This keeps
the encoding step consistent and fair across fusion strategies.

Three classifier families are trained and compared:

  1. MLP (deep)   — 2-layer feed-forward network on the concatenated vector.
                    Saved as: data/best_model_early_fusion_mlp.pt

  2. RF  (non-deep) — RandomForestClassifier.
                      Saved as: data/best_model_early_fusion_rf.pkl

  3. GBT (non-deep) — GradientBoostingClassifier.
                      Saved as: data/best_model_early_fusion_gbt.pkl

Selection metric: macro-F1 on the validation split (same as the existing model).
Evaluation: accuracy, macro-F1, per-class F1 on the test set.

Run
---
    cd long_covid_tweets
    python train_early_fusion.py --gpu 7
"""

import argparse
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
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",         type=int,   default=None)
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--batch-size",  type=int,   default=64,  help="Batch size for embedding extraction")
parser.add_argument("--clf-epochs",  type=int,   default=50,  help="Epochs for MLP classifier")
parser.add_argument("--clf-lr",      type=float, default=1e-3)
parser.add_argument("--max-len",     type=int,   default=128)
parser.add_argument("--rf-n-estimators",  type=int, default=300)
parser.add_argument("--gbt-n-estimators", type=int, default=200)
parser.add_argument("--gbt-max-depth",    type=int, default=5)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH  = "data/dataset.npz"
TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"
OUT_DIR    = "data"
VAL_FRAC   = 0.15
SEED       = args.seed

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
# Load raw data
# ---------------------------------------------------------------------------
data = np.load(DATA_PATH, allow_pickle=True)

X_train_static_all = data["X_train_static"].astype("float32")
X_test_static      = data["X_test_static"].astype("float32")
X_train_text_all   = data["X_train_text"].tolist()
X_test_text        = data["X_test_text"].tolist()
y_train_all        = data["y_train"].astype(int)
y_test             = data["y_test"].astype(int)
label_classes      = data["label_classes"].tolist()

N_CLASSES = len(label_classes)
D_TAB     = X_train_static_all.shape[1]
N_TRAIN   = len(y_train_all)
print(f"Classes ({N_CLASSES}): {label_classes}")
print(f"Tabular dim: {D_TAB}  |  Train: {N_TRAIN}  |  Test: {len(y_test)}")

# ---------------------------------------------------------------------------
# Validation split (identical to hparam_search.py)
# ---------------------------------------------------------------------------
rng_split = random.Random(SEED)
all_idx   = list(range(N_TRAIN))
rng_split.shuffle(all_idx)
n_val        = int(N_TRAIN * VAL_FRAC)
val_indices  = all_idx[:n_val]
train_indices = all_idx[n_val:]
print(f"Split — train: {len(train_indices)}  val: {len(val_indices)}")

# ---------------------------------------------------------------------------
# Extract frozen XLM-RoBERTa CLS embeddings  (Option A: pre-trained weights)
# ---------------------------------------------------------------------------
EMB_CACHE = os.path.join(OUT_DIR, "early_fusion_text_embeddings.pt")

if os.path.exists(EMB_CACHE):
    print(f"Loading cached embeddings from {EMB_CACHE}")
    cache = torch.load(EMB_CACHE, map_location="cpu")
    train_text_emb = cache["train"]
    test_text_emb  = cache["test"]
else:
    print(f"Extracting frozen XLM-RoBERTa embeddings (this may take a few minutes) …")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    text_enc  = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE)
    text_enc.eval()

    def encode_texts(texts, batch_size=args.batch_size):
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                max_length=args.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                out = text_enc(
                    input_ids=enc["input_ids"].to(DEVICE),
                    attention_mask=enc["attention_mask"].to(DEVICE),
                ).last_hidden_state[:, 0]            # CLS token
            all_emb.append(out.cpu())
            if (i // batch_size) % 20 == 0:
                print(f"  {i}/{len(texts)}", end="\r")
        return torch.cat(all_emb, dim=0)             # (n, 768)

    train_text_emb = encode_texts(X_train_text_all)
    test_text_emb  = encode_texts(X_test_text)
    torch.save({"train": train_text_emb, "test": test_text_emb}, EMB_CACHE)
    print(f"\nEmbeddings saved to {EMB_CACHE}")
    del text_enc

# ---------------------------------------------------------------------------
# Build concatenated feature matrices
# ---------------------------------------------------------------------------
X_train_ef_all = np.concatenate(
    [train_text_emb.numpy(), X_train_static_all], axis=1
).astype("float32")
X_test_ef = np.concatenate(
    [test_text_emb.numpy(), X_test_static], axis=1
).astype("float32")

D_EF = X_train_ef_all.shape[1]
print(f"Early-fusion feature dim: {D_EF}  ({768} text + {D_TAB} tabular)")

# Validation split arrays
X_tr_ef  = X_train_ef_all[train_indices]
X_val_ef = X_train_ef_all[val_indices]
y_tr     = y_train_all[train_indices]
y_val    = y_train_all[val_indices]


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class EarlyFusionMLP(nn.Module):
    """2-layer MLP classifier over the concatenated early-fusion feature vector."""

    def __init__(self, d_in: int, n_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# 1. MLP (deep)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Training Early-Fusion MLP (deep)")
print("=" * 55)

tr_ds  = TensorDataset(
    torch.tensor(X_tr_ef),
    torch.tensor(y_tr, dtype=torch.long),
)
val_ds = TensorDataset(
    torch.tensor(X_val_ef),
    torch.tensor(y_val, dtype=torch.long),
)
tr_dl  = DataLoader(tr_ds,  batch_size=256, shuffle=True,  num_workers=0)
val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

mlp = EarlyFusionMLP(D_EF, N_CLASSES, hidden=256, dropout=0.3).to(DEVICE)
optimizer  = torch.optim.AdamW(mlp.parameters(), lr=args.clf_lr, weight_decay=1e-4)
criterion  = nn.CrossEntropyLoss()
best_val_f1, best_state = 0.0, None

print(f"{'Epoch':>6}  {'tr_loss':>8}  {'tr_f1':>7}  {'vl_f1':>7}")
print("-" * 35)

for epoch in range(1, args.clf_epochs + 1):
    mlp.train()
    tot_loss, preds_tr, labs_tr = 0.0, [], []
    for xb, yb in tr_dl:
        logits = mlp(xb.to(DEVICE))
        loss   = criterion(logits, yb.to(DEVICE))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item() * len(yb)
        preds_tr += logits.argmax(1).cpu().tolist()
        labs_tr  += yb.tolist()

    mlp.eval()
    preds_vl, labs_vl = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            preds_vl += mlp(xb.to(DEVICE)).argmax(1).cpu().tolist()
            labs_vl  += yb.tolist()

    tr_f1 = f1_score(labs_tr, preds_tr, average="macro", zero_division=0)
    vl_f1 = f1_score(labs_vl, preds_vl, average="macro", zero_division=0)
    print(f"{epoch:>6}  {tot_loss/len(y_tr):>8.4f}  {tr_f1:>7.4f}  {vl_f1:>7.4f}")

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        best_state  = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

print(f"\nBest val macro-F1 (MLP): {best_val_f1:.4f}")

# Save MLP
mlp_path = os.path.join(OUT_DIR, "best_model_early_fusion_mlp.pt")
torch.save(
    {"state_dict": best_state, "d_in": D_EF, "n_classes": N_CLASSES,
     "hidden": 256, "dropout": 0.3, "label_classes": label_classes},
    mlp_path,
)

# Test evaluation (MLP)
mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
mlp.eval()
X_test_t = torch.tensor(X_test_ef, device=DEVICE)
with torch.no_grad():
    logits_test = mlp(X_test_t)
    y_proba_mlp = torch.softmax(logits_test, dim=1).cpu().numpy().astype("float32")
y_pred_mlp = y_proba_mlp.argmax(1).astype(np.int32)

print("\nTest results — MLP:")
print(classification_report(y_test, y_pred_mlp, target_names=label_classes))

# ---------------------------------------------------------------------------
# 2. Random Forest  (non-deep)
# ---------------------------------------------------------------------------
print("=" * 55)
print("Training Early-Fusion RF (non-deep)")
print("=" * 55)

rf = RandomForestClassifier(
    n_estimators=args.rf_n_estimators,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
)
rf.fit(X_tr_ef, y_tr)
rf_val_f1 = f1_score(y_val, rf.predict(X_val_ef), average="macro", zero_division=0)
print(f"Val macro-F1 (RF): {rf_val_f1:.4f}")

rf_path = os.path.join(OUT_DIR, "best_model_early_fusion_rf.pkl")
with open(rf_path, "wb") as fh:
    pickle.dump({"model": rf, "label_classes": label_classes}, fh)

y_pred_rf  = rf.predict(X_test_ef).astype(np.int32)
y_proba_rf = rf.predict_proba(X_test_ef).astype("float32")
print("\nTest results — RF:")
print(classification_report(y_test, y_pred_rf, target_names=label_classes))

# ---------------------------------------------------------------------------
# 3. Gradient Boosting  (non-deep)
# ---------------------------------------------------------------------------
print("=" * 55)
print("Training Early-Fusion GBT (non-deep)")
print("=" * 55)

sw = compute_sample_weight("balanced", y_tr)
gbt = GradientBoostingClassifier(
    n_estimators=args.gbt_n_estimators,
    max_depth=args.gbt_max_depth,
    learning_rate=0.1,
    random_state=SEED,
)
gbt.fit(X_tr_ef, y_tr, sample_weight=sw)
gbt_val_f1 = f1_score(y_val, gbt.predict(X_val_ef), average="macro", zero_division=0)
print(f"Val macro-F1 (GBT): {gbt_val_f1:.4f}")

gbt_path = os.path.join(OUT_DIR, "best_model_early_fusion_gbt.pkl")
with open(gbt_path, "wb") as fh:
    pickle.dump({"model": gbt, "label_classes": label_classes}, fh)

y_pred_gbt  = gbt.predict(X_test_ef).astype(np.int32)
y_proba_gbt = gbt.predict_proba(X_test_ef).astype("float32")
print("\nTest results — GBT:")
print(classification_report(y_test, y_pred_gbt, target_names=label_classes))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Early-fusion summary")
print("=" * 55)
print(f"  {'Model':<10}  {'Val macro-F1':>13}  {'Test macro-F1':>14}")
print("  " + "-" * 41)
for name, vf1, yp in [
    ("MLP",  best_val_f1, y_pred_mlp),
    ("RF",   rf_val_f1,   y_pred_rf),
    ("GBT",  gbt_val_f1,  y_pred_gbt),
]:
    tf1 = f1_score(y_test, yp, average="macro", zero_division=0)
    print(f"  {name:<10}  {vf1:>13.4f}  {tf1:>14.4f}")

# Save arrays for the best non-deep model (highest val macro-F1)
best_nondp_name, best_nondp_vf1, best_nondp_pred, best_nondp_proba = max(
    [("rf",  rf_val_f1,  y_pred_rf,  y_proba_rf),
     ("gbt", gbt_val_f1, y_pred_gbt, y_proba_gbt)],
    key=lambda x: x[1],
)
print(f"\nBest non-deep: {best_nondp_name.upper()} (val F1={best_nondp_vf1:.4f})")

for tag, yp, ypr in [
    ("early_fusion_mlp", y_pred_mlp, y_proba_mlp),
    (f"early_fusion_{best_nondp_name}", best_nondp_pred, best_nondp_proba),
]:
    np.save(os.path.join(OUT_DIR, f"y_pred_{tag}.npy"),  yp.astype(np.int32))
    np.save(os.path.join(OUT_DIR, f"y_proba_{tag}.npy"), ypr)
    np.save(os.path.join(OUT_DIR, f"y_true_{tag}.npy"),  y_test.astype(np.int32))
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as fh:
        fh.write(classification_report(y_test, yp, target_names=label_classes))

print(f"\nSaved models to {OUT_DIR}/")
