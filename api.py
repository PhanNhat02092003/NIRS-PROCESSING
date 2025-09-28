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
from utils import *
import os
import os
from dotenv import load_dotenv
load_dotenv()
VEGETABLE_CLASSIFICATION_FOLDER = os.getenv("VEGETABLE_CLASSIFICATION_FOLDER")
VERIFY_SUBSTANCES_FOLDER = os.getenv("VERIFY_SUBSTANCES_FOLDER")
PREDICT_SUBSTANCES_CONCENTRATION_FOLDER = os.getenv("PREDICT_SUBSTANCES_CONCENTRATION_FOLDER")

app = FastAPI(
    title="Xử lý các tác vụ liên quan đến phổ NIR"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
vegetable_classification_model, vegetable_classification_mean, vegetable_classification_std, vegetable_classification_label_encoder = load_vegetable_classification_model(VEGETABLE_CLASSIFICATION_FOLDER)
verify_substances_models = load_verify_substances_models(VERIFY_SUBSTANCES_FOLDER)
predict_substances_concentration_models = load_predict_substances_concentration_models(PREDICT_SUBSTANCES_CONCENTRATION_FOLDER)

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
    - Format:
        - Cột ID: ID của mẫu
        - Cột w_i: bước sóng thứ i (i: 0 -> 2135)
- Output: JSON chứa danh sách kết quả phân loại.
    - Format:
```
{
    "predictions": [
        {
            "id": "<ID của mẫu>",
            "result": "Kết quả phân loại"
        },
        ...
    ]
}
```
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

    try:
        df = pd.read_csv(io.StringIO(text))
        ids = df['ID'].to_list()
        df = df.drop(columns=["ID"])

        X = df.values.astype(np.float32)

        X = (X - vegetable_classification_mean) / vegetable_classification_std
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = vegetable_classification_model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        labels = vegetable_classification_label_encoder.inverse_transform(preds)
        results = [
            {
                "id": id,
                "result": label
            } for id, label in zip(ids, labels)
        ]

        return JSONResponse(content={"predictions": results})
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Lỗi khi xử lý: {str(e)}")

@app.post(
    "/nir-processing/machine-1/verify-substances",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Xác định sự tồn tại của các hợp chất có trong rau củ quả sử dụng phổ NIR cho máy 1",
    description="""
Upload một file CSV và trả về kết quả.  
- Input: file CSV (multipart/form-data)
    - Format:
        - Cột ID: ID của sample
        - Cột w_i: bước sóng thứ i (i: 0 -> 2135)
- Output: JSON chứa danh sách kết quả.
    - Format:
```
{
    "predictions": [
        "id": "<ID của mẫu>",
        "result": {
            "<Tên chất>": "true/false",
        }
    ]
}
```
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

    try:
        df = pd.read_csv(io.StringIO(text))
        ids = df['ID'].to_list()
        df = df.drop(columns=["ID"])
        X = df.values.astype(np.float32)

        total_predictions = {}
        for substance in verify_substances_models.keys():
            if verify_substances_models[substance]:
                total_predictions[substance] = verify_substances_models.predict(X)
            else:
                total_predictions[substance] = [None] * len(X)
        
        results = [
            {
                "id": ids[idx],
                "result": {
                    substance: bool(total_predictions[substance][idx])
                    for substance in total_predictions.keys()
                }
            }
            for idx in range(len(ids))
        ]

        return JSONResponse(content={"predictions": results})
    
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Lỗi khi xử lý: {str(e)}")

@app.post(
    "/nir-processing/machine-1/predict-substances-concentration",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Dự đoán nồng độ các hợp chất có trong rau củ quả sử dụng phổ NIR cho máy 1",
    description="""
Upload một file CSV và trả về kết quả.  
- Input: file CSV (multipart/form-data)
    - Format:
        - Cột ID: ID của sample
        - Cột w_i: bước sóng thứ i (i: 0 -> 2135)
- Output: JSON chứa danh sách kết quả.
    - Format:
```
{
    "predictions": [
        "id": "<ID của mẫu>",
        "result": {
            "<Tên chất>": "Hàm lượng chất có trong mẫu/None",
        }
    ]
}
```
"""
)
async def predict_substances_concentration(
    file: UploadFile = File(..., description="File CSV cần upload")
) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File phải có đuôi .csv")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File rỗng")

    try:
        text = decode_bytes_try_encodings(content)
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Không thể giải mã file CSV (encoding không hỗ trợ)")

    try:
        df = pd.read_csv(io.StringIO(text))
        if "ID" not in df.columns:
            raise HTTPException(status_code=400, detail="File CSV thiếu cột ID")

        ids = df["ID"].to_list()
        X_df = df.drop(columns=["ID"]).astype(np.float32)

        results = []
        for idx, sample_id in enumerate(ids):
            sample_result = {"id": sample_id, "result": {}}

            for substance in verify_substances_models.keys():
                cls_model = verify_substances_models[substance]
                reg_model = predict_substances_concentration_models[substance]

                if cls_model is None:
                    sample_result["result"][substance] = None
                    continue

                # classification
                cls_pred = int(cls_model.predict(X_df.iloc[[idx]])[0])
                if cls_pred == 0:
                    sample_result["result"][substance] = None
                else:
                    if reg_model is not None:
                        reg_pred = float(reg_model.predict(X_df.iloc[[idx]])[0])
                        sample_result["result"][substance] = reg_pred
                    else:
                        sample_result["result"][substance] = None

            results.append(sample_result)

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý: {str(e)}")

