# Mẫu test-data đạt chuẩn (chính sách ngưỡng cân bằng)

Xuất bằng pipeline deploy thật (`deploy_utils.analyze_spectrum`) sau khi
chỉnh lại ngưỡng quyết định của Stage 1/2 từ Recall≥0.99 sang **điểm cân
bằng false-alarm/miss-rate** (target recall riêng từng máy: 0.45 cho
FLAMENIR, 0.46 cho OCEANFX — điểm mà tỷ lệ báo động giả và tỷ lệ bỏ sót ở
mức mẫu gần bằng nhau nhất trên tập test-data thật; xem lịch sử hội thoại
để biết chi tiết quá trình sweep).

Toàn bộ các dòng của `${DATASET_ROOT}/test-data/{FLAMENIR,OCEANFX}.csv`
được chạy qua pipeline đầy đủ (phân loại thực phẩm → Stage 1 phát hiện →
Stage 2 severity) và so với nhãn thật:

- **category_correct**: mã thực phẩm dự đoán == mã thực phẩm thật (từ cột
  category gốc của file test, ánh xạ qua `food_ids.json`).
- **total_wrong** (trên tổng 19 chất): (số chất bị sai có/không-có mặt)
  + (số chất **thực sự có mặt** bị sai severity). Một chất thực sự không
  có mặt và được đoán đúng là "không có" đóng góp 0; một chất thực sự có
  mặt, được phát hiện đúng *và* đoán đúng an toàn/vượt ngưỡng cũng đóng
  góp 0.

Một dòng **đạt chuẩn** (và được xuất file) khi `category_correct` VÀ
`total_wrong <= 3`.

## Vì sao không phải 100%

Một khảo sát đầy đủ (xem lịch sử hội thoại) cho thấy không có mức ngưỡng
nào tạo ra một tỷ lệ đáng kể các dòng đúng tuyệt đối cả 19 chất cùng lúc —
tỷ lệ đúng tuyệt đối cao nhất chỉ đạt ~0.02% bất kể chọn ngưỡng nào, vì để
19 phán đoán độc lập đều đúng trên 1 dòng cần độ chính xác từng chất
~99%+, cao hơn nhiều so với năng lực hiện tại của các mô hình trên tập
test ngoài thực tế (bị lệch phân phối so với dữ liệu train) — một vài
trong 19 chất có khả năng phân biệt yếu hơn hẳn các chất còn lại. Dung sai
`total_wrong<=3` là một tiêu chí nới lỏng có chủ đích cho "gần như đúng",
không phải khẳng định pipeline đáng tin cậy ở từng mẫu riêng lẻ.

## Kết quả

| | FLAMENIR | OCEANFX |
|---|---:|---:|
| Số dòng test (n) | 26.623 | 37.479 |
| Trung bình `total_wrong` trên TOÀN BỘ dòng | 11.64 / 19 | 11.43 / 19 |
| Độ chính xác category tổng thể (category dự đoán thật) | 61.3% | 52.9% |
| Số dòng có `total_wrong<=3` (bất kể category) | 1.886 (7.08%) | 2.170 (5.79%) |
| ...trong đó category cũng đúng | **1.490 (79.0%)** | **1.354 (62.4%)** |
| Số dòng đạt chuẩn (`total_wrong<=3` VÀ category đúng) | **1.490 (5.60%)** | **1.354 (3.61%)** |

**Trả lời câu hỏi "trong các mẫu sai ≤3/19 chất, phân loại thực phẩm có
đúng không?": phần lớn là có, nhưng không phải tất cả** — FLAMENIR 79.0%
các mẫu sai ít (≤3 chất) cũng có category đúng; OCEANFX thấp hơn, 62.4%.
Vì không phải 100%, bước xuất file áp dụng CẢ HAI điều kiện cùng lúc
(`total_wrong<=3` VÀ category đúng) thay vì giả định category luôn đúng.

## Quy ước đặt tên file và thư mục

Mỗi file xuất ra là 1 CSV gồm đúng 1 dòng, giữ nguyên toàn bộ cột như file
test-data gốc (`w_0..w_127`, `category`, `comp_<Chất>` x19, `metadata`,
`has_label`, `model`), đặt trong thư mục theo máy rồi theo **tên loại
thực phẩm thật** (ground truth, tra từ `food_ids.json`) của chính mẫu đó:

```
test/FLAMENIR/<Tên thực phẩm>/processed/FL_{idx:05d}.csv
test/OCEANFX/<Tên thực phẩm>/processed/OC_{idx:05d}.csv
```

`idx` là vị trí (0-based) của dòng đó trong file gốc
`${DATASET_ROOT}/test-data/{FLAMENIR,OCEANFX}.csv` (vd:
`test/FLAMENIR/Cải Thìa/processed/FL_09535.csv` là dòng 9535 của
`FLAMENIR.csv`, loại thực phẩm thật là Cải Thìa).

Mỗi thư mục loại thực phẩm có 2 thư mục con:
- `processed/`: CSV đã qua xử lý (định dạng pipeline, mỗi file 1 dòng) —
  nội dung mô tả ở trên.
- `raw/`: file phổ dạng `.txt` tương ứng 1-1 với file CSV cùng tên (vd
  `raw/FL_09535.txt` ứng với `processed/FL_09535.csv`) — **tái tạo lại**
  từ cột `metadata` + các cột `w_*` của chính dòng CSV đó, theo đúng định
  dạng xuất thô Ocean Optics (header metadata + khối
  `wavelength<TAB>intensity`), khớp với 2 file mẫu `test/FLAMENIR/sample.txt`
  / `test/OCEANFX/sample.txt`. Trục bước sóng (cột đầu mỗi dòng dữ liệu)
  lấy cố định từ 2 file mẫu đó (cùng 1 máy quang phổ → cùng 1 trục bước
  sóng cho mọi mẫu); cột cường độ lấy từ `w_0..w_N` của dòng CSV.

  Lưu ý: đây là **tái tạo**, không phải file gốc thật — không có sẵn file
  `.txt` gốc theo từng mẫu thực phẩm thật trong bản copy dataset ở máy
  này (chỉ có file dark-reference/calibration trong
  `spectrum-differencce/`). Trường `Date` trong header có thể bị cắt cụt
  (vd "Tue Sep 29 23" thay vì đầy đủ giờ:phút:giây) — đây là lỗi có sẵn từ
  khâu xử lý dữ liệu gốc tạo ra cột `metadata` trong test-data CSV (bị cắt
  tại dấu `:` đầu tiên), không phải lỗi phát sinh khi tái tạo.

### Phân bố theo loại thực phẩm

| Loại thực phẩm | FLAMENIR | OCEANFX |
|---|---:|---:|
| Cà Chua | – | 3 |
| Cà Rốt | 500 | 632 |
| Cải Thìa | 120 | 89 |
| Dưa Leo | 397 | 512 |
| Khổ Qua | 324 | 16 |
| Đậu Cove | 149 | 102 |
| **Tổng** | **1.490** | **1.354** |

Lưu ý: đây là phân bố loại thực phẩm của các mẫu **đạt chuẩn** (sau khi
lọc), không phải phân bố của toàn bộ tập test-data gốc — một số loại thực
phẩm (vd Xà Lách, Cải Bẹ Xanh, Mồng Tơi — nhóm rau lá dễ nhầm lẫn, xem
`LEAFY_GREENS` trong `deploy_utils.py`) không xuất hiện mẫu nào đạt chuẩn.

## Lưu ý

File smoke-test có sẵn trước đây (`test/FLAMENIR/sample.csv`,
`test/OCEANFX/sample.csv`, nhắc tới trong CLAUDE.md) không còn tồn tại
trên đĩa tại thời điểm soạn báo cáo này — không xác định được nguyên nhân
chính xác (không có script nào trong phiên làm việc ghi vào đường dẫn đó),
và vì `test/` không được git track nên không thể khôi phục qua git.
