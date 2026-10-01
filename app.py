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

# id -> (Vietnamese name, English gloss), for the Swagger description below.
# Ids match FOOD_NAME_TO_ID / DATASET_ROOT/food_ids.json (F01..F09).
FOOD_ID_LEGEND = {
    "F01": ("Xà Lách", "Lettuce"),
    "F02": ("Cải Bẹ Xanh", "Mustard greens"),
    "F03": ("Cải Thìa", "Bok choy"),
    "F04": ("Mồng Tơi", "Malabar spinach"),
    "F05": ("Cà Chua", "Tomato"),
    "F06": ("Cà Rốt", "Carrot"),
    "F07": ("Dưa Leo", "Cucumber"),
    "F08": ("Khổ Qua", "Bitter melon"),
    "F09": ("Đậu Cove", "Cowpea / green bean"),
}

# Ids match SUBSTANCE_NAME_TO_ID / DATASET_ROOT/pesticide_ids.json (P01..P19).
SUBSTANCE_ID_LEGEND = {
    "P01": "Thiamethoxam",
    "P02": "Permethrin",
    "P03": "Metalaxyl",
    "P04": "Azoxystrobin",
    "P05": "Difenoconazole",
    "P06": "Cypermethrin",
    "P07": "Cyhalothrin",
    "P08": "Chlorantraniliprol",
    "P09": "Emamectin benzoate",
    "P10": "Chlorothalonil",
    "P11": "Triadimefon",
    "P12": "Cyantraniliprole",
    "P13": "Flutolanil",
    "P14": "Indoxacarb",
    "P15": "Abamectin",
    "P16": "Propamocarb.HCL",
    "P17": "Imidaclopird",
    "P18": "Chlopyrifos Methyl",
    "P19": "Chlothianidin",
}

_food_rows = "\n".join(f"| `{fid}` | {vi} | {en} |" for fid, (vi, en) in FOOD_ID_LEGEND.items())
_substance_rows = "\n".join(f"| `{pid}` | {name} |" for pid, name in SUBSTANCE_ID_LEGEND.items())

ANALYZE_DESCRIPTION = f"""
Nhận một phổ NIR thô, trả về: loại rau củ quả (`category`), các thuốc trừ
sâu phát hiện được (`substances_detected`), và mức độ an toàn tổng thể
(`safe`) -- chỉ `true` khi mọi thuốc phát hiện được đều dưới ngưỡng MRL;
nếu không, `substances_over_threshold` liệt kê đúng những thuốc vượt
ngưỡng (luôn là tập con của `substances_detected`).

Mỗi kết quả phân loại là một object `{{"code": ..., "conf-score": ...}}`:
- `category.code`, mỗi phần tử trong `substances_detected` /
  `substances_over_threshold` dùng **id** (không phải tên tiếng Việt), tra
  theo hai bảng dưới đây.
- `conf-score` (0-1): với `category`, là tỉ lệ 5 mô hình fold đồng thuận.
  Với từng thuốc (Bước 1/2), **không phải xác suất gốc của mô hình** mà là
  xác suất đó sau khi tái tâm quanh ngưỡng quyết định riêng của thuốc đó
  (dịch theo logit): đúng 0,5 tại ngưỡng, càng vượt xa ngưỡng càng tiến về
  1. Lý do: ngưỡng Bước 1/2 được chọn thấp để đảm bảo Recall ≥ 0,9, nên nếu
  dùng thẳng xác suất gốc thì một kết quả dương tính rõ ràng vẫn có thể
  hiện ra một con số trông thấp (ví dụ 0,15); công thức này đảm bảo mọi
  thuốc xuất hiện trong `substances_detected` / `substances_over_threshold`
  luôn có `conf-score` ≥ 0,5.

### Mã loại rau củ quả (`category`)

| id | Tên tiếng Việt | English |
|---|---|---|
{_food_rows}

### Mã thuốc trừ sâu (`substances_detected` / `substances_over_threshold`)

| id | Tên hoạt chất |
|---|---|
{_substance_rows}
"""

@app.post(
    "/nir-processing/analyze",
    response_class=JSONResponse,
    tags=["CSV"],
    summary="Phân tích phổ NIR: loại rau củ quả, thuốc trừ sâu phát hiện được và mức độ an toàn",
    description=ANALYZE_DESCRIPTION,
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
