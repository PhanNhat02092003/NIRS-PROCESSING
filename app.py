from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
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

@app.post(
    "/nir-processing/category-classification",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân loại rau củ quả (Cà Chua, Cải Bẹ Xanh, Cải Thìa, Cà Rốt, Đậu Cove, Dưa Leo, Khổ Qua, Mồng Tơi, Xà Lách)",
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
        categories = infer_category_classification(spectra, machine)
        results = infer_substances_detection(spectra, machine, categories)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post(
    "/nir-processing/substances-prediction",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân loại mức độ an toàn (An toàn / Vượt ngưỡng) từng hợp chất phát hiện được trong rau củ quả sử dụng phổ NIR",
)
async def substances_prediction(request: NirsRequest) -> JSONResponse:
    spectra = np.array(request.spectrum, dtype=np.float32)
    machine = request.machine
    try:
        categories = infer_category_classification(spectra, machine)
        detected_substances = infer_substances_detection(spectra, machine, categories)
        results = infer_substances_severity(spectra, machine, categories, detected_substances)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9000,
    )
