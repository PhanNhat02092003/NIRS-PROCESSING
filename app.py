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
import uvicorn
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
vegetable_classification_model, vegetable_classification_mean, vegetable_classification_std, idx2label = load_vegetable_classification_model(VEGETABLE_CLASSIFICATION_FOLDER)
verify_substances_models = load_verify_substances_models(VERIFY_SUBSTANCES_FOLDER)
predict_substances_concentration_models = load_predict_substances_concentration_models(PREDICT_SUBSTANCES_CONCENTRATION_FOLDER)

@app.post(
    "/nir-processing/machine-1/vegetable-classification",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân loại rau củ quả (Cà Chua, Cải Bẹ Xanh, Cải Thìa, Carrot, Đậu Cô Ve, Dưa Leo, Khổ Qua, Mồng Tơi, Xà Lách) sử dụng phổ NIR cho máy 1",
    description="""
Upload một file CSV và trả về nhãn phân loại.  
- Input: Dữ liệu phổ NIR
    - Format:
```
[
    ["<giá trị float>", ...],
    ...
]
```
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
async def vegetable_classification(request: NirsData) -> JSONResponse:
    try:
        X = np.array(request.spectrum, dtype=np.float32)

        X = (X - vegetable_classification_mean) / vegetable_classification_std
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = vegetable_classification_model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        labels = [idx2label[p] for p in preds]

        return JSONResponse(content={"predictions": labels})
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Lỗi khi xử lý: {str(e)}")

@app.post(
    "/nir-processing/machine-1/verify-substances",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Xác định sự tồn tại của các hợp chất có trong rau củ quả sử dụng phổ NIR cho máy 1",
    description="""
Upload một file CSV và trả về kết quả.  
- Input: Dữ liệu phổ NIR
    - Format:
```
[
    ["<giá trị float>", ...],
    ...
]
```
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
async def verify_substances(request: NirsData) -> JSONResponse:
    try:
        X = np.array(request.spectrum, dtype=np.float32)

        total_predictions = {}
        for substance in verify_substances_models.keys():
            if verify_substances_models[substance]:
                total_predictions[substance] = verify_substances_models[substance].predict(X)
            else:
                total_predictions[substance] = [None] * len(X)
        
        results = [
            {
                substance: bool(total_predictions[substance][idx])
                for substance in total_predictions.keys()
            }
            for idx in range(len(X))
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
- Input: Dữ liệu phổ NIR
    - Format:
```
[
    ["<giá trị float>", ...],
    ...
]
```
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
async def predict_substances_concentration(request: NirsData) -> JSONResponse:
    try:
        X = pd.DataFrame(np.array(request.spectrum, dtype=np.float32))

        results = []
        for idx in range(len(X)):
            sample_result = {}

            for substance in verify_substances_models.keys():
                cls_model = verify_substances_models[substance]
                reg_model = predict_substances_concentration_models[substance]

                if cls_model is None:
                    sample_result[substance] = None
                    continue

                # classification
                cls_pred = int(cls_model.predict(X.iloc[[idx]])[0])
                if cls_pred == 0:
                    sample_result[substance] = None
                else:
                    if reg_model is not None:
                        reg_pred = float(reg_model.predict(X.iloc[[idx]])[0])
                        sample_result[substance] = reg_pred
                    else:
                        sample_result[substance] = None

            results.append(sample_result)

        return JSONResponse(content={"predictions": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=9000,
    )
