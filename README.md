# NIRS Processing

A REST API for Near-Infrared Spectroscopy (NIRS) data processing, built with FastAPI. It provides vegetable classification (GuidedDCNet), pesticide-substance detection, and safety-level classification (both SMART-NIR) using deep learning models.

## Features

- **Vegetable Classification** — Classify vegetables into 9 categories (GuidedDCNet)
- **Substance Detection** — Detect the presence/absence of 19 pesticide substances (SMART-NIR, Bước 1)
- **Safety Classification** — Classify each detected substance as An toàn (safe) or Vượt ngưỡng (over MRL) (SMART-NIR, Bước 2)

## Project Structure

```
├── app.py                  # FastAPI application & endpoints
├── utils.py                # Inference utilities & model loading
├── model/
│   ├── classification_model.py    # SMART-NIR neural network architecture
│   ├── guideddcnet_model.py       # GuidedDCNet (diffusion-based) architecture
│   └── smartnir_food_model.py     # SMART-NIR + food one-hot (Bước 1/2) + food-prior shrinkage
├── dataset/
│   └── preprocessing.py    # Savitzky-Golay + SNV preprocessing (must match training)
├── checkpoint/
│   ├── category_classification_guideddcnet/{FLAMENIR,OCEANFX}/checkpoint_fold{1..5}.pth
│   └── substance_regression/stage1_smartnir_os2/{FLAMENIR,OCEANFX}/{substance}/...
│   └── substance_severity_smartnir/{FLAMENIR,OCEANFX}/{substance}/...
├── data/
│   ├── category_classification_guideddcnet/{FLAMENIR,OCEANFX}/fold_{1..5}/{stats.npz,label_encoder.pkl}
│   └── substance_regression/stage1_smartnir_os2/{FLAMENIR,OCEANFX}/{substance}/...
│   └── substance_severity_smartnir/{FLAMENIR,OCEANFX}/{substance}/...
├── requirements.txt
├── Dockerfile
└── server.sh
```

Model paths in `utils.py` are hardcoded relative to the project root
(`checkpoint/...`, `data/...`) -- there is no `.env` configuration for
model locations; the `checkpoint/` and `data/` folders must simply be
present (they are committed to this repo).

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (optional, falls back to CPU)

## Getting Started

### Option 1: Run Locally

1. **Create a virtual environment and install dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

   Or with Conda:

   ```bash
   conda create -n nir_ai python=3.11 -y
   conda activate nir_ai
   pip install -r requirements.txt
   ```

2. **Start the server:**

   ```bash
   python3 app.py
   ```

   Or using the provided script:

   ```bash
   bash server.sh
   ```

   The server will start on `http://0.0.0.0:9000`.

### Option 2: Run with Docker

1. **Build the image:**

   ```bash
   docker build -t ghcr.io/huytuong010101/nir_ai:dev .
   ```

2. **Run the container:**

   ```bash
   docker run -p 9000:9000 ghcr.io/huytuong010101/nir_ai:dev
   ```

   With GPU support (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

   ```bash
   docker run --gpus all -p 9000:9000 ghcr.io/huytuong010101/nir_ai:dev
   ```

   The API will be available at `http://localhost:9000`.

## Quick Usage

**Classify a vegetable:**

```bash
curl -X POST http://localhost:9000/nir-processing/category-classification \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "FLAMENIR"}'
```

**Detect substances:**

```bash
curl -X POST http://localhost:9000/nir-processing/substances-detection \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "FLAMENIR"}'
```

**Classify safety level of detected substances:**

```bash
curl -X POST http://localhost:9000/nir-processing/substances-prediction \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "FLAMENIR"}'
```

## API Reference

Interactive docs (Swagger UI) are available at `http://localhost:9000/docs`
when the server is running.

### Request body (all endpoints)

| Field | Type | Description |
|---|---|---|
| `spectrum` | `float[][]` | Batch of spectra. Each inner array's length must match the machine's wavelength count: **128** for `FLAMENIR`, **2136** for `OCEANFX`. |
| `machine` | `"FLAMENIR" \| "OCEANFX"` | Which spectrometer the spectra came from -- selects which trained models to load. |

Spectra are automatically preprocessed (Savitzky-Golay smoothing + SNV
scatter correction, `dataset/preprocessing.py`) to match what the models
were trained on -- send raw spectral intensities, not pre-processed ones.
For `OCEANFX`, spectra are additionally averaged in groups of 8 neighbouring
wavelengths (2136 → 264 points) before every model, matching training.

Each endpoint returns one result per input spectrum, in the same order. On
failure (e.g. wrong spectrum length, or missing model files for a
substance/machine) the endpoint returns HTTP 500 with `{"error": "<message>"}`.

### `POST /nir-processing/category-classification`

Returns the predicted vegetable category for each spectrum (GuidedDCNet),
voted across the 5 K-Fold models.

```json
{"results": ["Cải Thìa"]}
```

9 possible categories: `Khổ Qua`, `Mồng Tơi`, `Cải Thìa`, `Cà Chua`, `Cải Bẹ Xanh`,
`Dưa Leo`, `Xà Lách`, `Đậu Cove`, `Cà Rốt`.

### `POST /nir-processing/substances-detection`

Predicts the vegetable category internally first (used as an extra
one-hot input, since MRL/detection depends on the food type), then returns
the list of substances detected as present (SMART-NIR, Bước 1) for each
spectrum: 5-fold ensemble of calibrated probabilities (mean across folds),
compared against the mean of the 5 folds' Recall≥0.9 decision thresholds.

```json
{"results": [["Thiamethoxam"]]}
```

### `POST /nir-processing/substances-prediction`

Runs category classification and detection internally first, then
classifies each detected substance as **An toàn** (safe) or **Vượt ngưỡng**
(over the food-specific MRL) (SMART-NIR, Bước 2, same 5-fold ensemble
protocol as detection, plus blending with each substance's empirical
per-food prior).

```json
{"results": [{"Thiamethoxam": "An toàn"}]}
```

19 substances tracked (in `pesticide_ids.json` order, P01–P19):
`Thiamethoxam`, `Permethrin`, `Metalaxyl`, `Azoxystrobin`, `Difenoconazole`,
`Cypermethrin`, `Cyhalothrin`, `Chlorantraniliprol`, `Emamectin benzoate`,
`Chlorothalonil`, `Triadimefon`, `Cyantraniliprole`, `Flutolanil`,
`Indoxacarb`, `Abamectin`, `Propamocarb.HCL`, `Imidaclopird`,
`Chlopyrifos Methyl`, `Chlothianidin`.
A substance is only ever returned by detection/prediction if it had enough
samples of both classes on that machine to be trainable (see
`checkpoint/substance_regression/stage1_smartnir_os2/<machine>/<substance>`
and `checkpoint/substance_severity_smartnir/<machine>/<substance>` --
substances missing that folder are never trained and never predicted).

## Dependencies

| Package | Version |
|---|---|
| FastAPI | 0.115.14 |
| Uvicorn | 0.35.0 |
| PyTorch | 2.8.0 (CUDA 12.8) |
| TorchVision | 0.23.0 |
| scikit-learn | 1.7.0 |
| SciPy | (latest) |
| pandas | 2.3.2 |
| NumPy | 2.3.1 |
