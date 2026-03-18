# cf-lib

**Nearest-neighbour multimodal counterfactual generation library.**

`cf-lib` generates counterfactual explanations for multimodal classifiers by
searching for the closest opposite-class training samples in configurable
distance spaces. It supports tabular, time-series, text, and image modalities,
and provides several combination strategies.

> If you use this library in your research, please cite the associated paper
> (see [CITATION.cff](CITATION.cff)).

---

## Installation

Clone the repository and install the core dependencies:

```bash
git clone https://github.com/FoxtrotRomeo/cf-lib.git
cd cf-lib
pip install .
```

For text encoder support (E5, BERT, Word2Vec) and Intermediate-Fusion:

```bash
pip install ".[text,model]"
```

For image encoder support (ResNet-50, EfficientNet-B0, CLIP):

```bash
pip install ".[image]"
# For CLIP: pip install git+https://github.com/openai/CLIP.git
```

> **Note:** the helper modules (`counterfactual_helpers.py`,
> `counterfactual_evaluation_helpers.py`) must remain at the repository root
> alongside the `cf_lib/` package. When running scripts (e.g. from
> `examples/`), do so from the repository root so Python can find them.

---

## Quickstart

```python
from cf_lib import MultimodalDataset, CounterfactualLibrary
from cf_lib.unimodal import TabularNN, TimeSeriesNN, TextNN, ImageNN
from cf_lib.multimodal import FrankensteinNN, CombinedNN

dataset = MultimodalDataset(
    y_train=y_train,                 # np.ndarray (n_train,)  — only required field
    # Optional modalities:
    X_train_static=X_train_static,   # np.ndarray (n_train, D_static)
    X_test_static=X_test_static,     # np.ndarray (n_test,  D_static)
    X_train_ts={"labs": arr_train_labs, "meds": arr_train_meds},
    X_test_ts= {"labs": arr_test_labs,  "meds": arr_test_meds},
    X_train_text=train_texts,
    X_test_text=test_texts,
    X_train_img=train_images,        # PIL list, ndarray, or pre-computed embeddings
    X_test_img=test_images,
)

lib = CounterfactualLibrary(
    generators={
        "Tabular":       TabularNN(k=50),
        "Labs TS":       TimeSeriesNN("labs", k=50),
        "Meds TS":       TimeSeriesNN("meds", k=50),
        "Image":         ImageNN(k=20, image_encoder="resnet50"),
        "Frankenstein":  FrankensteinNN(image_encoder="resnet50"),
        "Combined":      CombinedNN(image_encoder="resnet50"),
    }
)

results = lib.generate(dataset, sample_idx=3, target_value=0)
# results -> {"Tabular": [...], "Labs TS": [...], "Image": [...], ...}
```

---

## Available generators

### Unimodal

| Class | Description |
|---|---|
| `TabularNN` | Nearest neighbours in the static/tabular feature space |
| `TimeSeriesNN` | Nearest neighbours in a named time-series space (DTW / Euclidean / LCSS) |
| `LabsNN` | Alias for `TimeSeriesNN("labs")` |
| `MedsNN` | Alias for `TimeSeriesNN("meds")` |
| `TextNN` | Nearest neighbours in text space (E5, BERT, TF-IDF, Word2Vec, BLEU, ROUGE) |
| `ImageNN` | Nearest neighbours in image space (ResNet-50, EfficientNet-B0, CLIP ViT-B/32, or pre-computed embeddings) |

### Multimodal

| Class | Description |
|---|---|
| `EarlyFusionNN` | NN in the concatenated multimodal feature space (static + TS-mean + text + image) |
| `IntermediateFusionNN` | NN in the classifier's latent (penultimate-layer) space — requires a Keras model; supports image input |
| `FrankensteinNN` | Independent per-modality searches; assembled by combining the best neighbours from each |
| `CombinedNN` | Per-modality searches intersected by shared candidates, ranked by summed distance |

---

## Dataset format

`MultimodalDataset` is a dataclass. Only `y_train` is required; all other
modalities default to `None`. Modalities absent from the dataset are
automatically skipped by every generator.

```python
MultimodalDataset(
    y_train,                 # REQUIRED — np.ndarray (n_train,), integer labels
    X_train_static=None,     # np.ndarray (n_train, D_static)
    X_test_static=None,      # np.ndarray (n_test,  D_static)
    X_train_ts=None,         # dict[str, np.ndarray (n_train, T, D)]
    X_test_ts=None,          # dict[str, np.ndarray (n_test,  T, D)]
    X_train_tab=None,        # dict[str, np.ndarray (n_train, D)]  — named tabular blocks
    X_test_tab=None,         # dict[str, np.ndarray (n_test,  D)]
    X_train_text=None,       # array-like of strings, length n_train
    X_test_text=None,        # array-like of strings, length n_test
    X_train_img=None,        # PIL list | ndarray (n,H,W,C) | pre-computed (n,D)
    X_test_img=None,         # same format as X_train_img
    y_test=None,             # np.ndarray (n_test,), optional
)
```

**Image input formats accepted by `ImageNN` and multimodal generators:**
- **Pre-computed embeddings** — 2-D `np.ndarray` of shape `(n, D)`, or a list of
  1-D vectors. Used as-is (no encoding step). Auto-detected when `image_encoder="precomputed"`.
- **Raw images** — list of PIL `Image` objects, 3-D ndarray `(n, H, W)` (grayscale),
  or 4-D ndarray `(n, H, W, C)`. Encoded by the chosen backbone before search.
- **Custom callable** — pass `embed_fn=my_fn` to override backbone encoding entirely.

---

## Experiments

The `examples/` directory contains two scripts:

- **`ablation_factory_template.py`** — factory functions that load a
  `MultimodalDataset` from `.npy` files on disk, or generate a synthetic
  dataset for smoke-testing.
- **`run_distance_ablation.py`** — CLI runner that sweeps combinations of
  distance metrics across all modalities and saves results to JSON/pickle.

Run from the **repository root**:

```bash
# Smoke test with synthetic data (includes image embeddings)
python examples/run_distance_ablation.py \
  --factory ablation_factory_template:build_synthetic_dataset_and_model \
  --max-combinations 5 \
  --max-samples 3

# Real data with image encoder sweep
python examples/run_distance_ablation.py \
  --factory ablation_factory_template:build_dataset_and_model \
  --factory-kwargs-json '{"data_root": "/path/to/fold_0", "ts_modalities": ["labs", "meds"], "load_image": true, "image_encoder": "resnet50"}' \
  --image-encoders resnet50 \
  --image-distance-metrics cosine,euclidean \
  --output-dir ablation_runs \
  --max-samples 25
```

The factory functions return a dict with `"dataset"`, `"model"`,
`"text_backend_kwargs"`, and `"image_backend_kwargs"`. The runner uses
`"image_backend_kwargs"` to forward device/batch-size settings to `ImageNN`.
Encoders whose dependencies are missing (e.g. `clip_vit_b32` without the
`clip` package) are skipped automatically.

---

## Evaluation

`counterfactual_evaluation_helpers.py` computes four objectives per
candidate: `outcome`, `proximity`, `sparsity`, and `plausibility`.

- `compute_objectives(...)` now supports named `tabular_modalities`,
  `ts_modalities`, `text_modalities`, and `image_modalities`, so evaluation
  is no longer limited to a single tabular / TS / text input.
- The legacy flat arguments are still supported and are auto-promoted into
  the newer dict-based format for backward compatibility.
- Text objectives now use generic embedding callables instead of requiring
  only the older E5-specific path.
- Image objectives mirror text evaluation with embedding-based proximity,
  sparsity, and LOF plausibility.
- Plausibility normalizers and LOF references can be fitted on the target
  class only, which makes plausibility scores reflect how typical a
  candidate is within the desired counterfactual class.

`examples/run_distance_ablation.py` can also attach objective summaries to
each ablation row via `objectives_kwargs` or
`objectives_kwargs_factory(text_cfg, image_cfg)`.

---

## Dependencies

| Package | Role |
|---|---|
| `numpy` | Core array operations |
| `scikit-learn` | Pairwise distances, TF-IDF |
| `aeon` | DTW and other time-series distances |
| `transformers` + `torch` | E5 / BERT text encoders *(optional)* |
| `gensim` | Word2Vec text encoder *(optional)* |
| `rouge-score` + `nltk` | ROUGE / BLEU text distance metrics *(optional)* |
| `torch` + `torchvision` + `Pillow` | Image encoding with ResNet-50 / EfficientNet-B0 *(optional)* |
| `clip` (OpenAI) | CLIP ViT-B/32 image encoder *(optional)* |
| `keras` | Loading models for `IntermediateFusionNN` *(optional)* |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
Derivatives and redistributions must retain the copyright notice and cite the
original work (see [CITATION.cff](CITATION.cff)).
