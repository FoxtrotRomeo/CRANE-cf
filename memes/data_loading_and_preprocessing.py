# %% Imports
import re

from datasets import load_dataset
from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# %% Download dataset + image files
# snapshot_download pulls the full HuggingFace repo, including the img/ directory
repo_path = Path(snapshot_download(
    repo_id="neuralcatcher/hateful_memes",
    repo_type="dataset",
))
img_dir = repo_path / "img"
print("Dataset repo path:", repo_path)
print("Image directory  :", img_dir)
print("Images available :", sum(1 for _ in img_dir.glob("*.png")) if img_dir.exists() else 0)

# %% Load dataset splits
ds = load_dataset("neuralcatcher/hateful_memes")
for split, data in ds.items():
    print(f"  {split}: {data.num_rows} rows")

# %% Combine train + validation for downstream split
# Official test set labels are unreliable in this version, so we pool
# train (8500) + validation (1040) and do a stratified 80/20 split.
df_all = pd.concat([
    ds["train"].to_pandas(),
    ds["validation"].to_pandas(),
], ignore_index=True)
print(f"\nCombined: {len(df_all)} rows")
df_all["label"].value_counts()

# %% Stratified 80/20 split
df_train, df_test = train_test_split(
    df_all, test_size=0.2, random_state=42, stratify=df_all["label"]
)
df_train = df_train.copy().reset_index(drop=True)
df_test  = df_test.copy().reset_index(drop=True)
print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")
print("Train label distribution:\n", df_train["label"].value_counts())

# %% Label array (already 0/1 integers)
label_classes = np.array(["not_hateful", "hateful"])  # index 0 / 1
y_train = df_train["label"].values.astype(np.int64)
y_test  = df_test["label"].values.astype(np.int64)

# %% Helper: load a PIL image from img_dir given the path string in the dataset
def _load_pil(img_path_str: str) -> Image.Image:
    """Load from the snapshot repo's img/ directory; falls back to the raw path."""
    fname = Path(img_path_str).name          # e.g. "42953.png"
    full  = img_dir / fname
    if not full.exists():
        full = Path(img_path_str)            # absolute path fallback
    return Image.open(full).convert("RGB")

# %% Text arrays (HTML tags stripped, newlines normalised)
def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(text.split())

TEXT_COL = "text"
X_train_text = df_train[TEXT_COL].apply(_clean).to_numpy()
X_test_text  = df_test[TEXT_COL].apply(_clean).to_numpy()

# %% Load raw images and resize to 224×224 (uint8, HWC layout)
# Stored as raw pixels so any encoder can be applied downstream
# (by ImageNN during CF generation or by the classifier during training).
# Shape: (n, 224, 224, 3) uint8  →  ~1.4 GB uncompressed for ~9 500 images.
IMG_SIZE = 224

def _load_images(img_paths: pd.Series) -> np.ndarray:
    arrays, n = [], len(img_paths)
    for i, p in enumerate(img_paths):
        try:
            img = _load_pil(p).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arrays.append(np.array(img, dtype=np.uint8))
        except Exception:
            arrays.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
        if i % 500 == 0:
            print(f"  {i}/{n} images loaded …")
    return np.stack(arrays)

print("Loading train images …")
X_train_img = _load_images(df_train["img"])
print("Loading test images …")
X_test_img  = _load_images(df_test["img"])
print(f"Image array shape: {X_train_img.shape}  dtype: {X_train_img.dtype}")

# %% Final shapes
print("X_train_text:  ", X_train_text.shape)
print("X_train_img:   ", X_train_img.shape)
print("y_train:       ", y_train.shape)
print("Classes:", dict(enumerate(label_classes)))

# %% Save to disk
import os

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

np.savez_compressed(
    os.path.join(OUT_DIR, "dataset.npz"),
    X_train_text   = X_train_text,
    X_test_text    = X_test_text,
    X_train_img    = X_train_img,
    X_test_img     = X_test_img,
    y_train        = y_train,
    y_test         = y_test,
    label_classes  = label_classes,
)
print("Saved to", OUT_DIR + "/dataset.npz")

# To reload later:
# data = np.load("data/dataset.npz", allow_pickle=True)
# X_train_text   = data["X_train_text"]     # (n,) string array
# X_train_img    = data["X_train_img"]      # (n, 2048) float32
# y_train        = data["y_train"]          # (n,) int64
# %%
