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

cd main   # all commands below run from main/, not the repo root

# Food-category classification (5-fold CV); METHOD = smartnir | guideddcnet | xgboost | lightgbm
MACHINE=FLAMENIR METHOD=smartnir python food_classification.py
MACHINE=OCEANFX BIN=8 METHOD=smartnir python food_classification.py   # OCEANFX: average wavelengths in groups of 8

# Step 1: per-substance presence/absence detection; METHOD = xgboost | lightgbm | smartnir | guideddcnet
MACHINE=FLAMENIR METHOD=lightgbm python stage1_detection.py

# Step 2: per-substance safe / over-MRL classification; METHOD = xgboost | lightgbm | smartnir | guideddcnet
MACHINE=FLAMENIR METHOD=lightgbm python stage2_safety.py

# Earlier concentration regression (no longer part of the report); METHOD = stage2 | ebar | nirmacnet | xspecmamba
MACHINE=FLAMENIR METHOD=stage2 python regression.py
```

Set `MACHINE` (`FLAMENIR` or `OCEANFX`) and `METHOD` in the environment (see the docstring at the top of each script for the other options); `main/.env` provides `DATASET_ROOT`. Ensure `../all-dataset/Danang-NIR/{machine}/ALL.csv` (relative to `main/`) exists. A small smoke-test CSV per machine lives at `main/test/{machine}/sample.csv`. `main/get_results.ipynb` is used for ad hoc inspection of saved history/checkpoints after training.

## Repository layout

The repo root holds only `.gitignore`, `CLAUDE.md`, `requirements.txt` and two subtrees -- everything runnable lives in one of them:

- `main/`: the Danang pesticide pipeline that the report is about -- `food_classification.py`, `stage1_detection.py`, `stage2_safety.py`, `regression.py` (each merges several methods, selected by `METHOD`), plus `make_holdout_test.py`, `dedup_conflicting_spectra.py`, `evaluate_holdout_stage1.py`, `compare_baselines.py`, `model/`, `dataset/`, `reports/`, `results/`. Commands below assume `cwd=main/`.
- `additional/{Grainit,Mango,Rapeseed}/`: benchmarks on external NIR datasets (scripts, benchmark CSVs and their own checkpoint/history/data); see its README. `additional/model/` and `additional/dataset/` hold the model/dataset modules used only by these benchmarks (EBAR, NirMACNet, XSpecMamba, PLS, ResNet1D, SpectraFormer baselines, per-dataset multitask/quality/regression Dataset classes); these scripts add both `../main/` and `../` (the `additional/` directory itself) to `sys.path`, so `model`/`dataset` resolve as namespace packages merging `main/model|dataset/` (shared: classification_model, guideddcnet_model, classification_dataset, preprocessing) with `additional/model|dataset/`.

## Architecture

### Data Flow
- Input: CSV files at `../all-dataset/Danang-NIR/{machine}/ALL.csv` (relative to `main/`)
  - Wavelength columns prefixed with `w_` (e.g., `w_1`, `w_2`, ...)
  - A `category` column for classification
  - One column per substance (value = concentration, `-1` means absent)
- Output layout differs per engine:
  - **Classification** (`task="category_classification"`): normalization stats + label encoder per fold in `data/{task}/{machine}/fold_{n}/`, checkpoints in `checkpoint/{task}/{machine}/checkpoint_fold{n}.pth`, history/plots in `history/{task}/{machine}/`
  - **Stage 1** (`task="substance_regression"`, XGBoost): per-substance, per-fold `StandardScaler` in `data/{task}/stage1/{machine}/{substance}/`, models in `checkpoint/{task}/stage1/{machine}/{substance}/{substance}_fold_{n}.json`, history/plots in `history/{task}/stage1/{machine}/{substance}/`
  - **Stage 2** (`task="substance_regression"`, deep learning): per-substance, per-fold normalization `.npz` in `data/{task}/stage2/{machine}/{substance}/fold_{n}/`, checkpoints/history/plots under the matching `checkpoint|history/{task}/stage2/{machine}/{substance}/`

### Deployed category classification: 6-way + leafy-green disambiguator

The 9 food categories include 4 spectrally-near-identical leafy greens (Xà
Lách, Cải Bẹ Xanh, Cải Thìa, Mồng Tơi) that the main classifier cannot
reliably tell apart. The deployed pipeline (`deploy_utils.py`) therefore
splits this into two steps instead of one 9-way guess:
1. `food_classification_6class.py` trains GuidedDCNet on
   `build_6class_labels.py`'s `ALL_downsampled3_6class.csv` (CAP=25
   downsampled, group-tagged data -- ~8,000 rows/machine -- with the 4
   leafy greens merged into one `"Rau Lá"` label; `StratifiedGroupKFold`
   on `_group_id`, same leakage-safe protocol as
   `food_classification_hierarchical.py`; `GDC_BATCH_SIZE` overridden to
   256, down from the 3072 default tuned for the full ALL.csv) --
   `task="category_classification_6way_guideddcnet"`.
2. Whenever that model predicts `"Rau Lá"`, `train_leafy_pca_lda.py`'s
   PCA+LDA model (`checkpoint/leafy_pca_lda/{machine}/model.joblib`,
   trained on `../all-dataset/Danang-NIR/leafy-position-scan/{machine}/` scanned at the **"Lá 2" / inner-leaf-face**
   position, ~97% CV accuracy on OCEANFX / ~85% on FLAMENIR) resolves
   which of the 4 it actually is. **This requires leafy-green samples to
   be scanned at the "Lá 2" position in production** -- a different
   position was not validated for this model and may be far less
   accurate (see `reports/figs/leafy_positions/`).

The original 9-class checkpoints (`task="category_classification_guideddcnet"`)
are left untouched and still used by standalone baseline-comparison
scripts (`compare_baselines.py`, `evaluate_deployed_pipeline_smartnir_food.py`,
`food_classification_downsampled.py`) -- only the live pipeline
(`deploy_utils.py`) switched to the 6-way + PCA+LDA design.

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
| `food_classification.py` | Category classification | SMART-NIR / GuidedDCNet / XGBoost / LightGBM | StratifiedKFold-5 | Accuracy |
| `stage1_detection.py` | Substance presence (binary) | XGBoost / LightGBM / SMART-NIR / GuidedDCNet | StratifiedKFold-5 | Accuracy, PR-AUC |
| `regression.py` (`METHOD=stage2`) | Substance concentration (earlier) | `SMARTNIRRegressor` (+ EBAR, NirMACNet, XSpecMamba) | KFold-5 | R² |

Stage 1 uses XGBoost with StandardScaler (scaler saved via `joblib`); Stage 2 uses the deep learning model with per-fold normalization saved as `.npz`. Both stage 1 and 2 loop over all 19 substances independently (one model per substance per fold) and skip a substance (while still creating its output folders) when there aren't enough valid samples:
- Stage 1 additionally undersamples the majority class per substance before running CV, so each substance is trained on a balanced (50/50) subset.
- Stage 2 skips a substance if it has zero or only one unique non-`-1` value (`RegressionNIRSDataset` raises `ValueError` in that case).

Note the classification and regression engines instantiate `SmartNIR*Config` with `d_model=128, depth=3, n_heads=4`, overriding the dataclass defaults (`d_model=64, depth=6, n_heads=8`) defined in `model/*.py`.

### Dataset Classes

- `ClassificationNIRSDataset`: Must call `.fit_normalization_and_labels(train_indices, save_dir)` before use
- `RegressionNIRSDataset`: Must call `.fit_normalization(train_indices, save_dir)` before use; automatically filters out rows where the target substance is `-1`; raises `ValueError` if no valid samples exist

### Substances Tracked (19 total)
IDs and iteration order come from `{DATASET_ROOT}/pesticide_ids.json` (P01–P19), which stage1/stage2 scripts read via `pesticide_name_to_id.keys()` — this is **not** alphabetical and does not match any thematic grouping, so don't assume a different order when a training log seems to "skip" a substance; check `pesticide_ids.json` before suspecting a bug.

| ID | Substance | ID | Substance |
|----|-----------|----|-----------|
| P01 | Thiamethoxam | P11 | Triadimefon |
| P02 | Permethrin | P12 | Cyantraniliprole |
| P03 | Metalaxyl | P13 | Flutolanil |
| P04 | Azoxystrobin | P14 | Indoxacarb |
| P05 | Difenoconazole | P15 | Abamectin |
| P06 | Cypermethrin | P16 | Propamocarb.HCL |
| P07 | Cyhalothrin | P17 | Imidaclopird |
| P08 | Chlorantraniliprol | P18 | Chlopyrifos Methyl |
| P09 | Emamectin benzoate | P19 | Chlothianidin |
| P10 | Chlorothalonil | | |
