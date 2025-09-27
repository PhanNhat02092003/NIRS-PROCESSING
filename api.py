# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import csv
from typing import List
import pandas as pd
import numpy as np
import joblib
import torch
from model.classification_model import *

app = FastAPI(
    title="Xử lý các tác vụ liên quan đến phổ NIR",
    description="API cho phép upload file CSV và trả về kết quả dưới dạng chuỗi",
)

# --- Bật CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SMARTNIR(cfg=SmartNIRClassificationConfig(
    signal_len=2136,
    out_ch_per_branch=128,
    d_model=256,
    depth=6,
    n_heads=8,
    classifier="kan",
    num_classes=9
)).to(device)

model.load_state_dict(torch.load("pretrained/smart_nir_classification_1.pth", map_location=device))
model.to(device)
model.eval()



def decode_bytes_try_encodings(content: bytes, encodings=("utf-8-sig", "utf-8", "latin-1")) -> str:
    for enc in encodings:
        try:
            return content.decode(enc)
        except Exception:
            continue
    raise UnicodeDecodeError("Unable to decode bytes with tried encodings", b"", 0, 1, "encoding error")

@app.post(
    "/nir-processing/machine-1/vegetable-classification",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân loại rau củ quả (Cà Chua, Cải Bẹ Xanh, Cải Thìa, Carrot, Đậu Cô Ve, Dưa Leo, Khổ Qua, Mồng Tơi, Xà Lách) sử dụng phổ NIR cho máy 1",
    description="""
Upload một file CSV và trả về nhãn phân loại.  
- Input: file CSV (multipart/form-data)  
- Output: JSON chứa danh sách kết quả phân loại.  
"""
)
async def vegetable_classification(file: UploadFile = File(..., description="File CSV cần upload")) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File phải có đuôi .csv")

    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"detail": "File rỗng"})

    try:
        text = decode_bytes_try_encodings(content)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Không thể giải mã file CSV (encoding không hỗ trợ)")

    stats = np.load("data/classification_data/dataset_1/train/stats.npz")
    mean, std = stats["mean"], stats["std"]
    label_encoder = joblib.load("data/classification_data/dataset_1/train/label_encoder.pkl")

    df = pd.read_csv(io.StringIO(text), index_col=0)
    X = df.values.astype(np.float32)

    X = (X - mean) / std
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # decode thành tên nhãn
    labels = label_encoder.inverse_transform(preds)

    return JSONResponse(content={"predictions": labels.tolist()})

@app.post(
    "/nir-processing/machine-1/verify-substances",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Xác định sự tồn tại của các hợp chất có trong rau củ quả sử dụng phổ NIR cho máy 1",
    description="""
Upload một file CSV và trả về nhãn xác nhận.  
- Input: file CSV (multipart/form-data)  
- Output: JSON chứa danh sách kết quả.  
"""
)
async def verify_substances(file: UploadFile = File(..., description="File CSV cần upload")) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File phải có đuôi .csv")

    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"detail": "File rỗng"})

    try:
        text = decode_bytes_try_encodings(content)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Không thể giải mã file CSV (encoding không hỗ trợ)")

    stats = np.load("data/classification_data/dataset_1/train/stats.npz")
    mean, std = stats["mean"], stats["std"]
    label_encoder = joblib.load("data/classification_data/dataset_1/train/label_encoder.pkl")

    df = pd.read_csv(io.StringIO(text), index_col=0)
    X = df.values.astype(np.float32)

    X = (X - mean) / std
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    labels = label_encoder.inverse_transform(preds)

    return JSONResponse(content={"predictions": labels.tolist()})



