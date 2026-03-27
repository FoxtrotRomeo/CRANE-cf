"""
Utilities for precomputing and reusing multimodal encoder embeddings.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as tvm
from transformers import AutoModel, AutoTokenizer

TEXT_ENCODERS = {
    "word2vec": {
        "feature_dim": 300,
    },
    "bert": {
        "model_name": "bert-base-uncased",
        "pooling": "cls",
        "normalize": False,
        "feature_dim": 768,
    },
    "minilm": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "pooling": "mean",
        "normalize": True,
        "feature_dim": 384,
    },
}

IMAGE_ENCODERS = {
    "resnet50": {
        "feature_dim": 2048,
    },
    "efficientnet_b0": {
        "feature_dim": 1280,
    },
    "vit_b_16": {
        "feature_dim": 768,
    },
}

DEFAULT_CACHE_PATH = os.path.join("data", "precomputed_embeddings.pt")
_WORD2VEC_MODELS = {}


class RawImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform):
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.fromarray(self.images[idx])
        return self.transform(img)


class EmbeddingDataset(Dataset):
    def __init__(self, text_features: torch.Tensor, image_features: torch.Tensor, labels: torch.Tensor):
        self.text_features = text_features
        self.image_features = image_features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "text_features": self.text_features[idx],
            "image_features": self.image_features[idx],
            "label": self.labels[idx],
        }


class RawMultimodalDataset(Dataset):
    def __init__(self, images: np.ndarray, texts: List[str], labels: torch.Tensor,
                 text_encoder_name: str, text_backend, image_transform, max_len: int):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.text_encoder_name = text_encoder_name
        self.text_backend = text_backend
        self.image_transform = image_transform
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.text_encoder_name == "word2vec":
            encoded = encode_word2vec_ids(self.texts[idx], self.text_backend, self.max_len)
        else:
            encoded = self.text_backend(
                self.texts[idx],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
        image = Image.fromarray(self.images[idx])
        return {
            "input_ids": encoded["input_ids"].squeeze(0) if encoded["input_ids"].ndim > 1 else encoded["input_ids"],
            "attention_mask": (
                encoded["attention_mask"].squeeze(0)
                if encoded["attention_mask"].ndim > 1 else encoded["attention_mask"]
            ),
            "image": self.image_transform(image),
            "label": self.labels[idx],
        }


class PrecomputedMultimodalClassifier(nn.Module):
    def __init__(self, text_input_dim: int, img_input_dim: int, n_classes: int,
                 text_hidden_dim: int, img_hidden_dim: int, dropout: float):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, img_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(text_hidden_dim + img_hidden_dim, n_classes)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_proj(text_features)
        img_emb = self.img_proj(image_features)
        return self.classifier(torch.cat([text_emb, img_emb], dim=1))


class FineTuneMultimodalClassifier(nn.Module):
    def __init__(self, text_encoder_name: str, image_encoder_name: str, n_classes: int,
                 text_hidden_dim: int, img_hidden_dim: int, dropout: float, device: str,
                 word2vec_path: str | None = None,
                 word2vec_vocab_tokens: List[str] | None = None):
        super().__init__()
        self.text_encoder_name = text_encoder_name
        if text_encoder_name == "word2vec":
            if word2vec_vocab_tokens is None:
                raise ValueError("word2vec_vocab_tokens is required for the word2vec encoder.")
            self.text_encoder = build_word2vec_embedding_layer(word2vec_path, word2vec_vocab_tokens)
            self.text_cfg = None
        else:
            _, self.text_encoder, self.text_cfg = build_text_backend(
                text_encoder_name, device, word2vec_path=word2vec_path
            )
        self.image_encoder, _ = build_image_backend(image_encoder_name, device)
        self.text_encoder.train()
        self.image_encoder.train()
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        for param in self.image_encoder.parameters():
            param.requires_grad = True

        text_input_dim, img_input_dim = get_feature_dims(text_encoder_name, image_encoder_name)
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_input_dim, img_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(text_hidden_dim + img_hidden_dim, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                images: torch.Tensor) -> torch.Tensor:
        if self.text_encoder_name == "word2vec":
            text_features = self.text_encoder(input_ids, attention_mask)
        else:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            text_features = pool_text_features(
                text_outputs,
                attention_mask,
                pooling=self.text_cfg["pooling"],
                normalize=self.text_cfg["normalize"],
            )
        image_features = self.image_encoder(images)
        text_emb = self.text_proj(text_features)
        img_emb = self.img_proj(image_features)
        return self.classifier(torch.cat([text_emb, img_emb], dim=1))


class Word2VecAverageEncoder(nn.Module):
    def __init__(self, embedding_weight: torch.Tensor):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_weight,
            freeze=False,
            padding_idx=0,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).type_as(embedded)
        summed = (embedded * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", str(text).lower())


def load_word2vec_model(word2vec_path: str):
    if word2vec_path is None:
        raise ValueError("word2vec_path is required when using the word2vec encoder.")
    if word2vec_path not in _WORD2VEC_MODELS:
        from gensim.models import KeyedVectors

        if not os.path.exists(word2vec_path):
            raise FileNotFoundError(f"word2vec file not found: {word2vec_path}")
        if word2vec_path.endswith(".kv"):
            model = KeyedVectors.load(word2vec_path)
        else:
            model = KeyedVectors.load_word2vec_format(
                word2vec_path,
                binary=word2vec_path.endswith(".bin"),
            )
        _WORD2VEC_MODELS[word2vec_path] = model
    return _WORD2VEC_MODELS[word2vec_path]


def collect_word2vec_tokens(texts: Iterable[str], word2vec_path: str) -> List[str]:
    kv = load_word2vec_model(word2vec_path)
    wv = getattr(kv, "wv", kv)
    seen = {}
    for text in texts:
        for token in _tokenize_text(text):
            if token in wv and token not in seen:
                seen[token] = None
    return list(seen.keys())


def build_word2vec_vocab(word2vec_path: str, vocab_tokens: List[str]) -> dict:
    kv = load_word2vec_model(word2vec_path)
    wv = getattr(kv, "wv", kv)
    stoi = {token: idx + 1 for idx, token in enumerate(vocab_tokens)}
    return {
        "stoi": stoi,
        "vocab_tokens": vocab_tokens,
        "vector_size": int(wv.vector_size),
        "word2vec_path": word2vec_path,
    }


def build_word2vec_embedding_layer(word2vec_path: str | None, vocab_tokens: List[str]):
    kv = load_word2vec_model(word2vec_path)
    wv = getattr(kv, "wv", kv)
    weight = np.zeros((len(vocab_tokens) + 1, int(wv.vector_size)), dtype=np.float32)
    for idx, token in enumerate(vocab_tokens, start=1):
        weight[idx] = np.asarray(wv[token], dtype=np.float32)
    return Word2VecAverageEncoder(torch.tensor(weight, dtype=torch.float32))


def encode_word2vec_ids(text: str, backend: dict, max_len: int) -> dict:
    tokens = _tokenize_text(text)[:max_len]
    ids = [backend["stoi"].get(tok, 0) for tok in tokens]
    mask = [1 if token_id != 0 else 0 for token_id in ids]
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids.extend([0] * pad_len)
        mask.extend([0] * pad_len)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def build_text_backend(name: str, device: str, word2vec_path: str | None = None):
    if name == "word2vec":
        return None, None, TEXT_ENCODERS[name]
    cfg = TEXT_ENCODERS[name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModel.from_pretrained(cfg["model_name"]).to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model, cfg


def build_text_tokenizer(name: str, word2vec_path: str | None = None,
                         vocab_tokens: List[str] | None = None):
    if name == "word2vec":
        if vocab_tokens is None:
            raise ValueError("vocab_tokens is required for the word2vec text encoder.")
        return build_word2vec_vocab(word2vec_path, vocab_tokens)
    return AutoTokenizer.from_pretrained(TEXT_ENCODERS[name]["model_name"])


def build_image_backend(name: str, device: str):
    if name == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT
        model = tvm.resnet50(weights=weights)
        model.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.DEFAULT
        model = tvm.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
    elif name == "vit_b_16":
        weights = tvm.ViT_B_16_Weights.DEFAULT
        model = tvm.vit_b_16(weights=weights)
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Unknown image encoder: {name}")

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, weights.transforms()


def get_image_transform(name: str):
    if name == "resnet50":
        return tvm.ResNet50_Weights.DEFAULT.transforms()
    if name == "efficientnet_b0":
        return tvm.EfficientNet_B0_Weights.DEFAULT.transforms()
    if name == "vit_b_16":
        return tvm.ViT_B_16_Weights.DEFAULT.transforms()
    raise ValueError(f"Unknown image encoder: {name}")


def pool_text_features(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor,
                       *, pooling: str, normalize: bool) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    if pooling == "cls":
        features = last_hidden_state[:, 0]
    elif pooling == "mean":
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        features = summed / denom
    else:
        raise ValueError(f"Unknown pooling strategy: {pooling}")

    if normalize:
        features = F.normalize(features, p=2, dim=1)
    return features


def _encode_texts(texts: List[str], encoder_name: str, device: str, batch_size: int,
                  max_len: int, word2vec_path: str | None = None) -> torch.Tensor:
    if encoder_name == "word2vec":
        kv = load_word2vec_model(word2vec_path)
        wv = getattr(kv, "wv", kv)
        out = np.zeros((len(texts), int(wv.vector_size)), dtype=np.float32)
        for i, text in enumerate(texts):
            vecs = [np.asarray(wv[tok], dtype=np.float32) for tok in _tokenize_text(text) if tok in wv]
            if vecs:
                out[i] = np.mean(np.vstack(vecs), axis=0)
        return torch.tensor(out, dtype=torch.float32)

    tokenizer, model, cfg = build_text_backend(encoder_name, device)
    batches: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded).last_hidden_state
            features = pool_text_features(
                outputs,
                encoded["attention_mask"],
                pooling=cfg["pooling"],
                normalize=cfg["normalize"],
            )

            batches.append(features.cpu().to(torch.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(batches, dim=0)


def _encode_images(images: np.ndarray, encoder_name: str, device: str, batch_size: int,
                   num_workers: int) -> torch.Tensor:
    model, transform = build_image_backend(encoder_name, device)
    loader = DataLoader(
        RawImageDataset(images, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    batches: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            features = model(batch.to(device))
            batches.append(features.cpu().to(torch.float32))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(batches, dim=0)


def _init_cache() -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    return {
        "schema_version": 2,
        "text": {},
        "image": {},
    }


def _cache_complete(cache: dict, modality: str, encoder_name: str) -> bool:
    if encoder_name not in cache.get(modality, {}):
        return False
    splits = cache[modality][encoder_name]
    return "train" in splits and "test" in splits


def _populate_missing_embeddings(cache: dict, *, X_train_text: List[str], X_test_text: List[str],
                                 X_train_img: np.ndarray, X_test_img: np.ndarray, device: str,
                                 batch_size: int, max_len: int, num_workers: int,
                                 word2vec_path: str | None) -> dict:
    for encoder_name in TEXT_ENCODERS:
        if _cache_complete(cache, "text", encoder_name):
            continue
        print(f"Precomputing text embeddings with {encoder_name} ...")
        cache["text"][encoder_name] = {
            "train": _encode_texts(
                X_train_text, encoder_name, device, batch_size, max_len, word2vec_path=word2vec_path
            ),
            "test": _encode_texts(
                X_test_text, encoder_name, device, batch_size, max_len, word2vec_path=word2vec_path
            ),
        }

    for encoder_name in IMAGE_ENCODERS:
        if _cache_complete(cache, "image", encoder_name):
            continue
        print(f"Precomputing image embeddings with {encoder_name} ...")
        cache["image"][encoder_name] = {
            "train": _encode_images(X_train_img, encoder_name, device, batch_size, num_workers),
            "test": _encode_images(X_test_img, encoder_name, device, batch_size, num_workers),
        }

    return cache


def load_or_precompute_embeddings(*, X_train_text: Iterable[str], X_test_text: Iterable[str],
                                  X_train_img: np.ndarray, X_test_img: np.ndarray, device: str,
                                  batch_size: int, max_len: int, num_workers: int = 2,
                                  cache_path: str = DEFAULT_CACHE_PATH,
                                  word2vec_path: str | None = None) -> dict:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading embedding cache from {cache_path} ...")
        cache = torch.load(cache_path, map_location="cpu")
    else:
        cache = _init_cache()

    if cache.get("schema_version") != 2:
        cache = _init_cache()

    cache = _populate_missing_embeddings(
        cache,
        X_train_text=list(X_train_text),
        X_test_text=list(X_test_text),
        X_train_img=X_train_img,
        X_test_img=X_test_img,
        device=device,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
        word2vec_path=word2vec_path,
    )
    torch.save(cache, cache_path)
    print(f"Embedding cache ready at {cache_path}")
    return cache


def get_feature_dims(text_encoder: str, image_encoder: str) -> Tuple[int, int]:
    text_dim = TEXT_ENCODERS[text_encoder]["feature_dim"]
    img_dim = IMAGE_ENCODERS[image_encoder]["feature_dim"]
    return text_dim, img_dim
