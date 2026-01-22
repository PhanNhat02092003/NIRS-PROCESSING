from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
from utils import *
import uvicorn

app = FastAPI(title="Xử lý các tác vụ liên quan đến phổ NIR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post(
    "/nir-processing/category-classification",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân loại rau củ quả (Cà Chua, Cải Bẹ Xanh, Cải Thìa, Carrot, Đậu Cô Ve, Dưa Leo, Khổ Qua, Mồng Tơi, Xà Lách)",
)
async def category_classification(request: NirsRequest) -> JSONResponse:
    spectra = np.array(request.spectrum, dtype=np.float32)
    machine = request.machine
    try:
        results = infer_category_classification(spectra, machine)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post(
    "/nir-processing/substances-detection",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phát hiện các hợp chất có trong rau củ quả sử dụng phổ NIR",
)
async def substances_detection(request: NirsRequest) -> JSONResponse:
    spectra = np.array(request.spectrum, dtype=np.float32)
    machine = request.machine
    try:
        results = infer_substances_detection(spectra, machine)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post(
    "/nir-processing/substances-prediction",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Dự đoán hàm lượng các hợp chất có trong rau củ quả sử dụng phổ NIR",
)
async def substances_prediction(request: NirsRequest) -> JSONResponse:
    spectra = np.array(request.spectrum, dtype=np.float32)
    machine = request.machine
    try:
        detected_substances = infer_substances_detection(spectra, machine)
        results = infer_substances_prediction(spectra, machine, detected_substances)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9000,
    )
