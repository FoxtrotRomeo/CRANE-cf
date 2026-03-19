"""Download word2vec-google-news-300 via gensim and save as .kv for fast reuse.

Usage:
    python real_or_fake_jobs/download_word2vec.py
    # → writes real_or_fake_jobs/data/word2vec_google_news_300.kv (~3.6 GB)

Then pass to the ablation:
    python run_cf_ablation.py --word2vec-path data/word2vec_google_news_300.kv
"""
from __future__ import annotations

import sys
from pathlib import Path

# Resolve output path relative to this script's directory
_DATA_DIR = Path(__file__).parent / "data"
_OUT_PATH = _DATA_DIR / "word2vec_google_news_300.kv"

if _OUT_PATH.exists():
    print(f"Already exists: {_OUT_PATH}  ({_OUT_PATH.stat().st_size / 1e9:.1f} GB)")
    sys.exit(0)

import gensim.downloader as api

print("Downloading word2vec-google-news-300 via gensim (≈1.7 GB) …")
kv = api.load("word2vec-google-news-300")

print(f"Saving to {_OUT_PATH} …")
_DATA_DIR.mkdir(parents=True, exist_ok=True)
kv.save(str(_OUT_PATH))
print(f"Done — {_OUT_PATH.stat().st_size / 1e9:.1f} GB, {len(kv):,} tokens, {kv.vector_size}-dim")
print()
print("Run the ablation with:")
print(f"  python run_cf_ablation.py --word2vec-path data/word2vec_google_news_300.kv")
