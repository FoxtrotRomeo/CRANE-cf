"""
Early-fusion classifier for Real-or-Fake Jobs.

Fusion strategy: each modality is first encoded into a fixed-size vector using
a frozen pre-trained encoder; the vectors are concatenated and fed directly to
a classifier.

  description CLS     (768-dim)  ╮
  company_profile CLS (768-dim)  ├─ concat → [3×768 ‖ tabular]
  requirements CLS    (768-dim)  ╯
  tabular features    (D_TAB-dim)

Total early-fusion feature dim: 3×768 + D_TAB

Encoding (Option A): CLS embeddings are extracted from the frozen *pre-trained*
DistilBERT weights, not from the fine-tuned classifier checkpoint.

NOTE: the existing MultimodalClassifier (hparam_search.py) is an
*intermediate-fusion* model.  This script adds the early-fusion counterpart.

Three classifier families are trained and compared:

  1. MLP (deep)   — 3-layer MLP on the concatenated vector.
                    Saved as: data/best_model_early_fusion_mlp.pt

  2. RF  (non-deep) — RandomForestClassifier.
                      Saved as: data/best_model_early_fusion_rf.pkl

  3. GBT (non-deep) — GradientBoostingClassifier.
                      Saved as: data/best_model_early_fusion_gbt.pkl

Selection metric: macro-F1 (class imbalance: ~5 % fake).
Evaluation: accuracy, macro-F1, per-class F1 on the test set.

Note on GBT speed: GradientBoostingClassifier on a ~2 300-dim × 14 k-sample
matrix may take 20–40 min.  Reduce --gbt-n-estimators if needed.

Run
---
    cd real_or_fake_jobs
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
parser.add_argument("--gpu",               type=int,   default=None)
parser.add_argument("--seed",              type=int,   default=42)
parser.add_argument("--max-len",           type=int,   default=512)
parser.add_argument("--embed-batch-size",  type=int,   default=32)
parser.add_argument("--clf-epochs",        type=int,   default=50)
parser.add_argument("--clf-lr",            type=float, default=1e-3)
parser.add_argument("--rf-n-estimators",   type=int,   default=300)
parser.add_argument("--gbt-n-estimators",  type=int,   default=100,
                    help="Keep low (100) — GBT on 2300-dim is slow.")
parser.add_argument("--gbt-max-depth",     type=int,   default=4)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH  = "data/dataset.npz"
TEXT_MODEL = "distilbert-base-uncased"
OUT_DIR    = "data"
VAL_FRAC   = 0.15
SEED       = args.seed
TEXT_FIELDS = ("description", "company_profile", "requirements")

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

X_train_static_all  = data["X_train_static"].astype("float32")
X_test_static       = data["X_test_static"].astype("float32")
X_train_desc_all    = data["X_train_description"].tolist()
X_test_desc         = data["X_test_description"].tolist()
X_train_profile_all = data["X_train_company_profile"].tolist()
X_test_profile      = data["X_test_company_profile"].tolist()
X_train_reqs_all    = data["X_train_requirements"].tolist()
X_test_reqs         = data["X_test_requirements"].tolist()
y_train_all         = data["y_train"].astype(int)
y_test              = data["y_test"].astype(int)
label_classes       = data["label_classes"].tolist()

N_CLASSES = len(label_classes)
D_TAB     = X_train_static_all.shape[1]
N_TRAIN   = len(y_train_all)
print(f"Classes ({N_CLASSES}): {label_classes}")
print(f"D_TAB={D_TAB}  |  N_train={N_TRAIN}  |  N_test={len(y_test)}")

# Validation split (same as hparam_search.py)
rng_split = random.Random(SEED)
all_idx   = list(range(N_TRAIN))
rng_split.shuffle(all_idx)
n_val         = int(N_TRAIN * VAL_FRAC)
val_indices   = all_idx[:n_val]
train_indices = all_idx[n_val:]
print(f"Split — train: {len(train_indices)}  val: {len(val_indices)}")

# ---------------------------------------------------------------------------
# Extract frozen DistilBERT CLS embeddings per field  (Option A)
# ---------------------------------------------------------------------------
EMB_CACHE = os.path.join(OUT_DIR, "early_fusion_text_embeddings.pt")

if os.path.exists(EMB_CACHE):
    print(f"Loading cached CLS embeddings from {EMB_CACHE}")
    cache = torch.load(EMB_CACHE, map_location="cpu")
    train_embs = cache["train"]   # dict: field → (N_TRAIN, 768)
    test_embs  = cache["test"]
else:
    print("Extracting frozen DistilBERT CLS embeddings for all three text fields …")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
    enc       = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE)
    enc.eval()

    def _encode(texts):
        out = []
        bs  = args.embed_batch_size
        for i in range(0, len(texts), bs):
            tok = tokenizer(
                texts[i:i+bs],
                max_length=args.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                cls = enc(
                    input_ids=tok["input_ids"].to(DEVICE),
                    attention_mask=tok["attention_mask"].to(DEVICE),
                ).last_hidden_state[:, 0]
            out.append(cls.cpu())
            if (i // bs) % 20 == 0:
                print(f"  {i}/{len(texts)}", end="\r")
        print()
        return torch.cat(out, dim=0)  # (n, 768)

    train_embs, test_embs = {}, {}
    for field, tr_texts, te_texts in [
        ("description",     X_train_desc_all,    X_test_desc),
        ("company_profile", X_train_profile_all, X_test_profile),
        ("requirements",    X_train_reqs_all,    X_test_reqs),
    ]:
        print(f"\nEncoding {field} …")
        train_embs[field] = _encode(tr_texts)
        test_embs[field]  = _encode(te_texts)

    torch.save({"train": train_embs, "test": test_embs}, EMB_CACHE)
    print(f"Saved embeddings to {EMB_CACHE}")
    del enc

# ---------------------------------------------------------------------------
# Build concatenated feature matrices
# ---------------------------------------------------------------------------
# Concat order: description ‖ company_profile ‖ requirements ‖ tabular
def _build_ef(embs_dict, static, indices=None):
    parts = [embs_dict[f].numpy() for f in TEXT_FIELDS] + [static]
    X = np.concatenate(parts, axis=1).astype("float32")
    return X[indices] if indices is not None else X

X_train_ef_all = _build_ef(train_embs, X_train_static_all)
X_test_ef      = _build_ef(test_embs,  X_test_static)
D_EF = X_train_ef_all.shape[1]
print(f"\nEarly-fusion feature dim: {D_EF}  (3×768={3*768} text + {D_TAB} tabular)")

X_tr_ef  = X_train_ef_all[train_indices]
X_val_ef = X_train_ef_all[val_indices]
y_tr     = y_train_all[train_indices]
y_val    = y_train_all[val_indices]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class EarlyFusionMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),      nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//4, n_classes),
        )
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 1. MLP (deep)
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Training Early-Fusion MLP (deep)")
print("=" * 55)

tr_ds  = TensorDataset(torch.tensor(X_tr_ef),  torch.tensor(y_tr,  dtype=torch.long))
val_ds = TensorDataset(torch.tensor(X_val_ef), torch.tensor(y_val, dtype=torch.long))
tr_dl  = DataLoader(tr_ds,  batch_size=128, shuffle=True,  num_workers=0)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

mlp   = EarlyFusionMLP(D_EF, N_CLASSES, hidden=512, dropout=0.3).to(DEVICE)
opt   = torch.optim.AdamW(mlp.parameters(), lr=args.clf_lr, weight_decay=1e-4)
crit  = nn.CrossEntropyLoss()
best_mlp_vf1, best_mlp_state = 0.0, None

print(f"{'Epoch':>6}  {'tr_loss':>8}  {'vl_f1':>7}")
print("-" * 26)
for epoch in range(1, args.clf_epochs + 1):
    mlp.train()
    tot, pp, ll = 0.0, [], []
    for xb, yb in tr_dl:
        logits = mlp(xb.to(DEVICE))
        loss   = crit(logits, yb.to(DEVICE))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(yb)
        pp  += logits.argmax(1).cpu().tolist()
        ll  += yb.tolist()
    mlp.eval()
    pv, lv = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            pv += mlp(xb.to(DEVICE)).argmax(1).cpu().tolist()
            lv += yb.tolist()
    vf1 = f1_score(lv, pv, average="macro", zero_division=0)
    print(f"{epoch:>6}  {tot/len(y_tr):>8.4f}  {vf1:>7.4f}")
    if vf1 > best_mlp_vf1:
        best_mlp_vf1   = vf1
        best_mlp_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}

print(f"\nBest val macro-F1 (MLP): {best_mlp_vf1:.4f}")
mlp_path = os.path.join(OUT_DIR, "best_model_early_fusion_mlp.pt")
torch.save({"state_dict": best_mlp_state, "d_in": D_EF, "n_classes": N_CLASSES,
            "hidden": 512, "dropout": 0.3, "label_classes": label_classes}, mlp_path)

mlp.load_state_dict({k: v.to(DEVICE) for k, v in best_mlp_state.items()})
mlp.eval()
with torch.no_grad():
    logits_test = mlp(torch.tensor(X_test_ef, device=DEVICE))
y_proba_mlp = torch.softmax(logits_test, dim=1).cpu().numpy().astype("float32")
y_pred_mlp  = y_proba_mlp.argmax(1).astype(np.int32)
print("\nTest — MLP:")
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
rf_vf1 = f1_score(y_val, rf.predict(X_val_ef), average="macro", zero_division=0)
print(f"Val macro-F1 (RF): {rf_vf1:.4f}")
rf_path = os.path.join(OUT_DIR, "best_model_early_fusion_rf.pkl")
with open(rf_path, "wb") as fh:
    pickle.dump({"model": rf, "label_classes": label_classes}, fh)
y_pred_rf  = rf.predict(X_test_ef).astype(np.int32)
y_proba_rf = rf.predict_proba(X_test_ef).astype("float32")
print("\nTest — RF:")
print(classification_report(y_test, y_pred_rf, target_names=label_classes))

# ---------------------------------------------------------------------------
# 3. Gradient Boosting  (non-deep)
# ---------------------------------------------------------------------------
print("=" * 55)
print(f"Training Early-Fusion GBT (non-deep, n_estimators={args.gbt_n_estimators})")
print("=" * 55)

sw  = compute_sample_weight("balanced", y_tr)
gbt = GradientBoostingClassifier(
    n_estimators=args.gbt_n_estimators,
    max_depth=args.gbt_max_depth,
    learning_rate=0.1,
    random_state=SEED,
    verbose=1,          # show progress (can be slow on high-dim data)
)
gbt.fit(X_tr_ef, y_tr, sample_weight=sw)
gbt_vf1 = f1_score(y_val, gbt.predict(X_val_ef), average="macro", zero_division=0)
print(f"Val macro-F1 (GBT): {gbt_vf1:.4f}")
gbt_path = os.path.join(OUT_DIR, "best_model_early_fusion_gbt.pkl")
with open(gbt_path, "wb") as fh:
    pickle.dump({"model": gbt, "label_classes": label_classes}, fh)
y_pred_gbt  = gbt.predict(X_test_ef).astype(np.int32)
y_proba_gbt = gbt.predict_proba(X_test_ef).astype("float32")
print("\nTest — GBT:")
print(classification_report(y_test, y_pred_gbt, target_names=label_classes))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Early-fusion summary")
print("=" * 55)
print(f"  {'Model':<8}  {'Val macro-F1':>13}  {'Test macro-F1':>14}")
print("  " + "-" * 39)
for name, vf1, yp in [
    ("MLP", best_mlp_vf1, y_pred_mlp),
    ("RF",  rf_vf1,       y_pred_rf),
    ("GBT", gbt_vf1,      y_pred_gbt),
]:
    tf1 = f1_score(y_test, yp, average="macro", zero_division=0)
    print(f"  {name:<8}  {vf1:>13.4f}  {tf1:>14.4f}")

for tag, yp, ypr in [
    ("early_fusion_mlp", y_pred_mlp, y_proba_mlp),
    ("early_fusion_rf",  y_pred_rf,  y_proba_rf),
    ("early_fusion_gbt", y_pred_gbt, y_proba_gbt),
]:
    np.save(os.path.join(OUT_DIR, f"y_pred_{tag}.npy"),  yp)
    np.save(os.path.join(OUT_DIR, f"y_proba_{tag}.npy"), ypr)
    np.save(os.path.join(OUT_DIR, f"y_true_{tag}.npy"),  y_test.astype(np.int32))
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as fh:
        fh.write(classification_report(y_test, yp, target_names=label_classes))
print(f"\nSaved models and arrays to {OUT_DIR}/")
