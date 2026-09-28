# NIRS Processing

A REST API for Near-Infrared Spectroscopy (NIRS) data processing, built with FastAPI. A single endpoint takes a raw spectrum and returns the vegetable category (GuidedDCNet), which pesticide substances were detected, and an overall safety verdict (both SMART-NIR).

## Features

One endpoint, one spectrum in, three things out:
- **Vegetable category** — one of 9 categories (GuidedDCNet)
- **Detected pesticide substances** — presence/absence of the 19 tracked substances (SMART-NIR, Bước 1)
- **Safety verdict** — safe only if every detected substance is An toàn (under its food-specific MRL); otherwise lists which substance(s) are Vượt ngưỡng (SMART-NIR, Bước 2). This is always derived from the same per-substance detection result, so it can never name a substance that wasn't reported as detected.

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

```bash
curl -X POST http://localhost:9000/nir-processing/analyze \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "FLAMENIR"}'
```

## API Reference

Interactive docs (Swagger UI) are available at `http://localhost:9000/docs`
when the server is running.

### `POST /nir-processing/analyze`

| Field | Type | Description |
|---|---|---|
| `spectrum` | `float[][]` | Batch of spectra. Each inner array's length must match the machine's wavelength count: **128** for `FLAMENIR`, **2136** for `OCEANFX`. |
| `machine` | `"FLAMENIR" \| "OCEANFX"` | Which spectrometer the spectra came from -- selects which trained models to load. |

Spectra are automatically preprocessed (Savitzky-Golay smoothing + SNV
scatter correction, `dataset/preprocessing.py`) to match what the models
were trained on -- send raw spectral intensities, not pre-processed ones.
For `OCEANFX`, spectra are additionally averaged in groups of 8 neighbouring
wavelengths (2136 → 264 points) before every model, matching training.

Runs three models per spectrum, in order: vegetable category (GuidedDCNet,
5-fold majority vote) → substances detected (SMART-NIR Bước 1, using the
predicted category as an extra input, since MRL/detection is food-dependent;
5-fold ensemble of calibrated probabilities vs. the mean of the 5 folds'
Recall≥0.9 thresholds) → safety verdict of each detected substance (SMART-NIR
Bước 2, same ensembling plus blending with each substance's empirical
per-food prior). `safe` is true only if every detected substance came back
An toàn; `substances_over_threshold` is read off that same per-substance
result, so it is always a subset of `substances_detected`.

`category` and every substance id are returned as **ids** (`F01`..`F09`,
`P01`..`P19`), not names -- see the id tables below, which are also shown
in the endpoint's description on the Swagger UI (`/docs`) and in its
`description` field on `/openapi.json`.

```json
{
  "results": [
    {
      "category": "F03",
      "substances_detected": ["P01"],
      "safe": false,
      "substances_over_threshold": ["P01"]
    }
  ]
}
```

One result per input spectrum, in the same order. On failure (e.g. wrong
spectrum length, or missing model files for a substance/machine) the
endpoint returns HTTP 500 with `{"error": "<message>"}`.

#### Food category ids (`category`)

| id | Tên tiếng Việt | English | id | Tên tiếng Việt | English |
|---|---|---|---|---|---|
| `F01` | Xà Lách | Lettuce | `F06` | Cà Rốt | Carrot |
| `F02` | Cải Bẹ Xanh | Mustard greens | `F07` | Dưa Leo | Cucumber |
| `F03` | Cải Thìa | Bok choy | `F08` | Khổ Qua | Bitter melon |
| `F04` | Mồng Tơi | Malabar spinach | `F09` | Đậu Cove | Cowpea / green bean |
| `F05` | Cà Chua | Tomato | | | |

#### Pesticide substance ids (`substances_detected` / `substances_over_threshold`)

| id | Substance | id | Substance |
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

A substance is only ever returned as detected/over-threshold if it had
enough samples of both classes on that machine to be trainable (see
`checkpoint/substance_regression/stage1_smartnir_os2/<machine>/<substance>`
and `checkpoint/substance_severity_smartnir/<machine>/<substance>` --
substances missing that folder are never trained and never reported).

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
