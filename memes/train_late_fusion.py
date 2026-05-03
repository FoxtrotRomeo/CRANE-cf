"""
Late-fusion classifier for Hateful Memes.

Fusion strategy: text and image are each processed by an independent classifier
up to the probability stage; the final prediction is the average of the two
branch probabilities.

  Text branch  →  classifier_text(text_emb)   →  p_text   (2,)
  Image branch →  classifier_image(image_emb) →  p_image  (2,)
  Combined     →  (p_text + p_image) / 2

NOTE: the existing model (hparam_search.py) is an *intermediate-fusion* model.
This script adds the late-fusion counterpart.

For each of the 3 text encoders and 3 image encoders, independent branch
classifiers are trained.  The combination (text_enc × image_enc) with the
highest combined validation macro-F1 is selected as the best configuration.

Two classifier families are trained per branch:

  Deep:    2-layer MLP on the frozen pre-computed embedding.
  Non-deep: RandomForestClassifier  (class_weight=balanced)
           + GradientBoostingClassifier  (sample_weight balanced).
           Best selected by val macro-F1.

Saved as (best encoder combination per family):
  data/best_model_late_fusion_mlp_text.pt
  data/best_model_late_fusion_mlp_image.pt
  data/best_model_late_fusion_rf_text.pkl
  data/best_model_late_fusion_rf_image.pkl   ← or gbt_, whichever is best
  data/best_model_late_fusion_gbt_text.pkl
  data/best_model_late_fusion_gbt_image.pkl

A summary CSV is written to data/late_fusion_results.csv.

Run
---
    cd memes
    python train_late_fusion.py --gpu 7
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
parser.add_argument("--gpu",              type=int,   default=None)
parser.add_argument("--seed",             type=int,   default=42)
parser.add_argument("--clf-epochs",       type=int,   default=50)
parser.add_argument("--clf-lr",           type=float, default=1e-3)
parser.add_argument("--rf-n-estimators",  type=int,   default=300)
parser.add_argument("--gbt-n-estimators", type=int,   default=200)
parser.add_argument("--gbt-max-depth",    type=int,   default=5)
parser.add_argument("--word2vec-path",    type=str,
                    default="../real_or_fake_jobs/data/word2vec_google_news_300.kv")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH   = "data/dataset.npz"
OUT_DIR     = "data"
VAL_FRAC    = 0.15
SEED        = args.seed
RESULTS_CSV = os.path.join(OUT_DIR, "late_fusion_results.csv")

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

word2vec_vocab = collect_word2vec_tokens(
    X_train_text_all + X_test_text, args.word2vec_path
)

# Validation split
rng_split = random.Random(SEED)
all_idx   = list(range(N_TRAIN))
rng_split.shuffle(all_idx)
n_val         = int(N_TRAIN * VAL_FRAC)
val_indices   = all_idx[:n_val]
train_indices = all_idx[n_val:]

y_tr  = y_train_all[train_indices].numpy().astype(int)
y_val = y_train_all[val_indices].numpy().astype(int)
print(f"Split — train: {len(y_tr)}  val: {len(y_val)}")

# ---------------------------------------------------------------------------
# Load precomputed embeddings
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
# Branch classifier helpers
# ---------------------------------------------------------------------------
class BranchMLP(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x):
        return self.net(x)


def train_branch_mlp(X_tr, y_tr, X_val, y_val, d_in, epochs, lr):
    tr_ds  = TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                            torch.tensor(y_tr,  dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                            torch.tensor(y_val, dtype=torch.long))
    tr_dl  = DataLoader(tr_ds,  batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False)

    mlp  = BranchMLP(d_in, N_CLASSES, hidden=128, dropout=0.3).to(DEVICE)
    opt  = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_vf1, best_state = 0.0, None
    for _ in range(epochs):
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


def predict_mlp(mlp, state, X, d_in):
    mlp.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(X, dtype=torch.float32, device=DEVICE))
    return torch.softmax(logits, dim=1).cpu().numpy().astype("float32")


def train_branch_nd(X_tr, y_tr, X_val, y_val):
    """Train RF + GBT; return (best_vf1, best_model, best_name, rf, gbt)."""
    sw  = compute_sample_weight("balanced", y_tr)
    rf  = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        class_weight="balanced", n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr)
    rf_vf1 = f1_score(y_val, rf.predict(X_val), average="macro", zero_division=0)

    gbt = GradientBoostingClassifier(
        n_estimators=args.gbt_n_estimators,
        max_depth=args.gbt_max_depth,
        learning_rate=0.1, random_state=SEED,
    )
    gbt.fit(X_tr, y_tr, sample_weight=sw)
    gbt_vf1 = f1_score(y_val, gbt.predict(X_val), average="macro", zero_division=0)

    if rf_vf1 >= gbt_vf1:
        return rf_vf1, rf, "rf", rf, gbt
    else:
        return gbt_vf1, gbt, "gbt", rf, gbt


# ---------------------------------------------------------------------------
# Sweep all text × image encoder combinations
# ---------------------------------------------------------------------------
csv_rows = []

# Track best per combined (text_enc, image_enc) × family
best_combo = {
    "mlp":   {"combined_vf1": -1.0},
    "nondp": {"combined_vf1": -1.0},
}

# Per-encoder branch scores (to pick best individual encoder per modality)
text_branch_scores_mlp  = {}   # enc → best val f1
text_branch_scores_nd   = {}
image_branch_scores_mlp = {}
image_branch_scores_nd  = {}

# Branch models cache (avoid re-training the same encoder)
text_mlp_cache  = {}   # enc → (vf1, state, mlp, d_in, test_proba)
text_nd_cache   = {}   # enc → (vf1, best_model, name, rf, gbt, test_proba)
image_mlp_cache = {}
image_nd_cache  = {}

for t_enc in TEXT_ENCODERS:
    if t_enc not in text_mlp_cache:
        t_tr   = embedding_cache["text"][t_enc]["train"][train_indices].numpy().astype("float32")
        t_val  = embedding_cache["text"][t_enc]["train"][val_indices].numpy().astype("float32")
        t_test = embedding_cache["text"][t_enc]["test"].numpy().astype("float32")
        d_t    = t_tr.shape[1]

        print(f"\nText encoder: {t_enc}  d={d_t}")
        vf1_t_mlp, state_t_mlp, mlp_t = train_branch_mlp(
            t_tr, y_tr, t_val, y_val, d_t, args.clf_epochs, args.clf_lr
        )
        proba_t_mlp_test = predict_mlp(mlp_t, state_t_mlp, t_test, d_t)
        text_mlp_cache[t_enc] = (vf1_t_mlp, state_t_mlp, mlp_t, d_t, proba_t_mlp_test)
        print(f"  Text MLP  val F1: {vf1_t_mlp:.4f}")

        vf1_t_nd, best_t_nd, nd_name, rf_t, gbt_t = train_branch_nd(
            t_tr, y_tr, t_val, y_val
        )
        proba_t_nd_test = best_t_nd.predict_proba(t_test).astype("float32")
        text_nd_cache[t_enc] = (vf1_t_nd, best_t_nd, nd_name, rf_t, gbt_t, proba_t_nd_test)
        print(f"  Text ND ({nd_name}) val F1: {vf1_t_nd:.4f}")

for i_enc in IMAGE_ENCODERS:
    if i_enc not in image_mlp_cache:
        i_tr   = embedding_cache["image"][i_enc]["train"][train_indices].numpy().astype("float32")
        i_val  = embedding_cache["image"][i_enc]["train"][val_indices].numpy().astype("float32")
        i_test = embedding_cache["image"][i_enc]["test"].numpy().astype("float32")
        d_i    = i_tr.shape[1]

        print(f"\nImage encoder: {i_enc}  d={d_i}")
        vf1_i_mlp, state_i_mlp, mlp_i = train_branch_mlp(
            i_tr, y_tr, i_val, y_val, d_i, args.clf_epochs, args.clf_lr
        )
        proba_i_mlp_test = predict_mlp(mlp_i, state_i_mlp, i_test, d_i)
        image_mlp_cache[i_enc] = (vf1_i_mlp, state_i_mlp, mlp_i, d_i, proba_i_mlp_test)
        print(f"  Image MLP  val F1: {vf1_i_mlp:.4f}")

        vf1_i_nd, best_i_nd, nd_name, rf_i, gbt_i = train_branch_nd(
            i_tr, y_tr, i_val, y_val
        )
        proba_i_nd_test = best_i_nd.predict_proba(i_test).astype("float32")
        image_nd_cache[i_enc] = (vf1_i_nd, best_i_nd, nd_name, rf_i, gbt_i, proba_i_nd_test)
        print(f"  Image ND ({nd_name}) val F1: {vf1_i_nd:.4f}")

# Evaluate all combinations on validation set
for t_enc in TEXT_ENCODERS:
    vf1_t_mlp, state_t, _, _, proba_t_mlp_test = text_mlp_cache[t_enc]
    vf1_t_nd,  bm_t_nd, nd_t_name, _, _, proba_t_nd_test = text_nd_cache[t_enc]
    t_val_proba_mlp = predict_mlp(
        text_mlp_cache[t_enc][2], state_t,
        embedding_cache["text"][t_enc]["train"][val_indices].numpy().astype("float32"),
        text_mlp_cache[t_enc][3],
    )
    t_val_proba_nd  = text_nd_cache[t_enc][1].predict_proba(
        embedding_cache["text"][t_enc]["train"][val_indices].numpy().astype("float32")
    ).astype("float32")

    for i_enc in IMAGE_ENCODERS:
        _, state_i, _, _, proba_i_mlp_test = image_mlp_cache[i_enc]
        vf1_i_nd,  bm_i_nd, nd_i_name, _, _, proba_i_nd_test = image_nd_cache[i_enc]
        i_val_proba_mlp = predict_mlp(
            image_mlp_cache[i_enc][2], state_i,
            embedding_cache["image"][i_enc]["train"][val_indices].numpy().astype("float32"),
            image_mlp_cache[i_enc][3],
        )
        i_val_proba_nd = image_nd_cache[i_enc][1].predict_proba(
            embedding_cache["image"][i_enc]["train"][val_indices].numpy().astype("float32")
        ).astype("float32")

        comb_mlp_val = f1_score(
            y_val, (0.5*t_val_proba_mlp + 0.5*i_val_proba_mlp).argmax(1),
            average="macro", zero_division=0,
        )
        comb_nd_val = f1_score(
            y_val, (0.5*t_val_proba_nd + 0.5*i_val_proba_nd).argmax(1),
            average="macro", zero_division=0,
        )

        csv_rows.append({
            "text_encoder": t_enc, "image_encoder": i_enc,
            "mlp_combined_val_f1": f"{comb_mlp_val:.6f}",
            "nd_combined_val_f1":  f"{comb_nd_val:.6f}",
        })

        if comb_mlp_val > best_combo["mlp"]["combined_vf1"]:
            best_combo["mlp"] = {
                "combined_vf1": comb_mlp_val,
                "t_enc": t_enc, "i_enc": i_enc,
                "t_state": state_t, "t_d": text_mlp_cache[t_enc][3],
                "i_state": state_i, "i_d": image_mlp_cache[i_enc][3],
                "test_proba_t": proba_t_mlp_test,
                "test_proba_i": proba_i_mlp_test,
            }
        if comb_nd_val > best_combo["nondp"]["combined_vf1"]:
            best_combo["nondp"] = {
                "combined_vf1": comb_nd_val,
                "t_enc": t_enc, "i_enc": i_enc,
                "t_model": bm_t_nd, "t_nd_name": nd_t_name,
                "i_model": bm_i_nd, "i_nd_name": nd_i_name,
                "t_rf": text_nd_cache[t_enc][3],   "t_gbt": text_nd_cache[t_enc][4],
                "i_rf": image_nd_cache[i_enc][3],  "i_gbt": image_nd_cache[i_enc][4],
                "test_proba_t": proba_t_nd_test,
                "test_proba_i": proba_i_nd_test,
            }

# ---------------------------------------------------------------------------
# Save best models per family
# ---------------------------------------------------------------------------
bm = best_combo["mlp"]
torch.save(
    {"state_dict": bm["t_state"], "d_in": bm["t_d"],
     "text_encoder": bm["t_enc"], "label_classes": label_classes},
    os.path.join(OUT_DIR, "best_model_late_fusion_mlp_text.pt"),
)
torch.save(
    {"state_dict": bm["i_state"], "d_in": bm["i_d"],
     "image_encoder": bm["i_enc"], "label_classes": label_classes},
    os.path.join(OUT_DIR, "best_model_late_fusion_mlp_image.pt"),
)
print(f"\nBest MLP combo: text={bm['t_enc']} image={bm['i_enc']}  val F1={bm['combined_vf1']:.4f}")

bnd = best_combo["nondp"]
for branch, model_key, name_key, rf_key, gbt_key, enc_key in [
    ("text",  "t_model", "t_nd_name", "t_rf", "t_gbt", "t_enc"),
    ("image", "i_model", "i_nd_name", "i_rf", "i_gbt", "i_enc"),
]:
    enc = bnd[enc_key]
    with open(os.path.join(OUT_DIR, f"best_model_late_fusion_rf_{branch}.pkl"), "wb") as fh:
        pickle.dump({"model": bnd[rf_key], "encoder": enc,
                     "label_classes": label_classes}, fh)
    with open(os.path.join(OUT_DIR, f"best_model_late_fusion_gbt_{branch}.pkl"), "wb") as fh:
        pickle.dump({"model": bnd[gbt_key], "encoder": enc,
                     "label_classes": label_classes}, fh)
print(f"Best ND  combo: text={bnd['t_enc']} image={bnd['i_enc']}  val F1={bnd['combined_vf1']:.4f}")

# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 55)
print("Test results")
print("=" * 55)

for family, info in [("mlp", bm), ("nondp", bnd)]:
    p_combined = 0.5 * info["test_proba_t"] + 0.5 * info["test_proba_i"]
    y_pred_t   = p_combined.argmax(1)
    print(f"\n[{family.upper()}]  text={info['t_enc']} image={info['i_enc']}")
    print(classification_report(y_test, y_pred_t, target_names=label_classes))
    tag = f"late_fusion_{family}"
    np.save(os.path.join(OUT_DIR, f"y_proba_{tag}.npy"), p_combined.astype("float32"))
    np.save(os.path.join(OUT_DIR, f"y_pred_{tag}.npy"),  y_pred_t.astype(np.int32))
    np.save(os.path.join(OUT_DIR, f"y_true_{tag}.npy"),  y_test.astype(np.int32))
    with open(os.path.join(OUT_DIR, f"classification_report_{tag}.txt"), "w") as fh:
        fh.write(f"Late-fusion [{family.upper()}]  "
                 f"text={info['t_enc']} image={info['i_enc']}\n\n")
        fh.write(classification_report(y_test, y_pred_t, target_names=label_classes))

# Write CSV
with open(RESULTS_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(
        fh, fieldnames=["text_encoder", "image_encoder",
                        "mlp_combined_val_f1", "nd_combined_val_f1"]
    )
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"\nFull results written to {RESULTS_CSV}")
