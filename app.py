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
    "/nir-processing/analyze",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân tích phổ NIR: loại rau củ quả, thuốc trừ sâu phát hiện được và mức độ an toàn",
)
async def analyze(request: NirsRequest) -> JSONResponse:
    spectra = np.array(request.spectrum, dtype=np.float32)
    machine = request.machine
    try:
        results = analyze_spectrum(spectra, machine)
        return JSONResponse(content={"results": results})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9000,
    )
