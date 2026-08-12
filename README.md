# NIRS Processing

A REST API for Near-Infrared Spectroscopy (NIRS) data processing, built with FastAPI. It provides vegetable classification, chemical substance detection, and substance concentration prediction using deep learning (SMARTNIR) and XGBoost models.

## Features

- **Vegetable Classification** — Classify vegetables into 9 categories (Tomato, Carrot, Cucumber, etc.)
- **Substance Detection** — Detect the presence/absence of 9 chemical compounds
- **Concentration Prediction** — Predict concentration levels of detected substances

## Project Structure

```
├── app.py                  # FastAPI application & endpoints
├── utils.py                # Inference utilities & model loading
├── model/
│   └── classification_model.py   # SMARTNIR neural network architecture
├── pretrained/
│   ├── vegetable_classification/  # SMARTNIR weights, labels, stats
│   ├── verify_substances/         # XGBoost binary classifiers (.pkl)
│   └── substances_concentration_prediction/  # XGBoost regressors (.pkl)
├── requirements.txt
├── Dockerfile
├── server.sh
└── API_DOCS.md             # Full API documentation
```

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

2. **Configure environment variables:**

   Create a `.env` file in the project root:

   ```env
   VEGETABLE_CLASSIFICATION_FOLDER=pretrained/vegetable_classification
   VERIFY_SUBSTANCES_FOLDER=pretrained/verify_substances
   PREDICT_SUBSTANCES_CONCENTRATION_FOLDER=pretrained/substances_concentration_prediction
   ```

3. **Start the server:**

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
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "machine-1"}'
```

**Detect substances:**

```bash
curl -X POST http://localhost:9000/nir-processing/substances-detection \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "machine-1"}'
```

**Predict concentrations:**

```bash
curl -X POST http://localhost:9000/nir-processing/substances-prediction \
  -H "Content-Type: application/json" \
  -d '{"spectrum": [[0.12, 0.45, 0.78, ...]], "machine": "machine-1"}'
```

## API Documentation

See [API_DOCS.md](API_DOCS.md) for the full API reference including request/response schemas, supported labels, and processing pipeline details.

Interactive docs are also available at `http://localhost:9000/docs` (Swagger UI) when the server is running.

## Dependencies

| Package | Version |
|---|---|
| FastAPI | 0.115.14 |
| Uvicorn | 0.35.0 |
| PyTorch | 2.8.0 (CUDA 12.8) |
| TorchVision | 0.23.0 |
| scikit-learn | 1.7.0 |
| XGBoost | 3.0.5 |
| pandas | 2.3.2 |
| NumPy | 2.3.1 |
