# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Near-Infrared Spectroscopy (NIRS) machine learning pipeline** for pesticide detection and quantification in agricultural samples. It processes NIR spectral data from two spectrometer machines (FLAMENIR and OCEANFX) to:
1. Classify samples by category (vegetable type)
2. Detect presence/absence of 19 pesticide substances (Stage 1 binary classification using XGBoost)
3. Quantify pesticide concentration in detected samples (Stage 2 regression using a deep learning model)

## Running the Training Scripts

```bash
# Install dependencies (requires CUDA 12.8 for GPU)
pip install -r requirements.txt

# Train the category classification model (5-fold CV, StratifiedKFold)
python classification_engine.py

# Train Stage 1: binary presence/absence detection per substance (XGBoost)
python regression_stage1_engine.py

# Train Stage 2: concentration regression per substance (deep learning)
python regression_stage2_engine.py
```

Before running, edit the `machine`/`task` variables inside each script's `if __name__ == "__main__":` block (set `machine` to `"FLAMENIR"` or `"OCEANFX"`), and ensure `../all-dataset/Danang-NIR/{machine}/ALL.csv` exists. A small smoke-test CSV per machine lives at `test/{machine}/sample.csv`. `get_results.ipynb` (root) is used for ad hoc inspection of saved history/checkpoints after training.

## Architecture

### Data Flow
- Input: CSV files at `../all-dataset/Danang-NIR/{machine}/ALL.csv`
  - Wavelength columns prefixed with `w_` (e.g., `w_1`, `w_2`, ...)
  - A `category` column for classification
  - One column per substance (value = concentration, `-1` means absent)
- Output layout differs per engine:
  - **Classification** (`task="category_classification"`): normalization stats + label encoder per fold in `data/{task}/{machine}/fold_{n}/`, checkpoints in `checkpoint/{task}/{machine}/checkpoint_fold{n}.pth`, history/plots in `history/{task}/{machine}/`
  - **Stage 1** (`task="substance_regression"`, XGBoost): per-substance, per-fold `StandardScaler` in `data/{task}/stage1/{machine}/{substance}/`, models in `checkpoint/{task}/stage1/{machine}/{substance}/{substance}_fold_{n}.json`, history/plots in `history/{task}/stage1/{machine}/{substance}/`
  - **Stage 2** (`task="substance_regression"`, deep learning): per-substance, per-fold normalization `.npz` in `data/{task}/stage2/{machine}/{substance}/fold_{n}/`, checkpoints/history/plots under the matching `checkpoint|history/{task}/stage2/{machine}/{substance}/`

### Model Architecture (SMARTNIRClassifier / SMARTNIRRegressor)

Both models share the same backbone defined in `model/classification_model.py` and `model/regression_model.py`:

1. **MultiKernelBlock**: Parallel Conv1d branches with 4 kernel sizes (4, 8, 16, 32), all stride=4, outputs concatenated → `4 * out_ch_per_branch` channels
2. **PatchProjector**: Linear projection + CLS token + learnable positional embeddings (ViT-style)
3. **TransformerEncoder**: Stack of `depth` EncoderBlocks with DualMLP (splits `d_model` in half, processes independently)
4. **Head**: Either `KANClassifier`/`KANRegressor` (custom Gaussian-RBF KAN layers) or standard `MLPClassifier`/`MLPRegressor`
   - Default is `"kan"` classifier

### Three Tasks / Three Engines

| Engine | Task | Model | CV Strategy | Key Metric |
|--------|------|-------|-------------|------------|
| `classification_engine.py` | Category classification | `SMARTNIRClassifier` | StratifiedKFold-5 | Accuracy |
| `regression_stage1_engine.py` | Substance presence (binary) | XGBoost (`binary:logistic`) | StratifiedKFold-5 | Accuracy |
| `regression_stage2_engine.py` | Substance concentration | `SMARTNIRRegressor` | KFold-5 | R² |

Stage 1 uses XGBoost with StandardScaler (scaler saved via `joblib`); Stage 2 uses the deep learning model with per-fold normalization saved as `.npz`. Both stage 1 and 2 loop over all 19 substances independently (one model per substance per fold) and skip a substance (while still creating its output folders) when there aren't enough valid samples:
- Stage 1 additionally undersamples the majority class per substance before running CV, so each substance is trained on a balanced (50/50) subset.
- Stage 2 skips a substance if it has zero or only one unique non-`-1` value (`RegressionNIRSDataset` raises `ValueError` in that case).

Note the classification and regression engines instantiate `SmartNIR*Config` with `d_model=128, depth=3, n_heads=4`, overriding the dataclass defaults (`d_model=64, depth=6, n_heads=8`) defined in `model/*.py`.

### Dataset Classes

- `ClassificationNIRSDataset`: Must call `.fit_normalization_and_labels(train_indices, save_dir)` before use
- `RegressionNIRSDataset`: Must call `.fit_normalization(train_indices, save_dir)` before use; automatically filters out rows where the target substance is `-1`; raises `ValueError` if no valid samples exist

### Substances Tracked (19 total)
`Thiamethoxam, Permethrin, Metalaxyl, Azoxystrobin, Imidaclopird, Difenoconazole, Cypermethrin, Cyhalothrin, Chlorantraniliprol, Chlopyrifos Methyl, Emamectin benzoate, Chlorothalonil, Triadimefon, Cyantraniliprole, Flutolanil, Indoxacarb, Abamectin, Propamocarb.HCL, Chlothianidin`
