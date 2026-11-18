# PCB Lab — Project học máy phát hiện lỗi mạch in

Ứng dụng cho đề tài **Phát hiện lỗi PCB dựa trên cặp ảnh mẫu chuẩn – ảnh kiểm tra**.
Đọc ảnh, tạo ảnh hiệu hoặc tensor hai kênh, huấn luyện YOLOv8n/YOLO11n/Faster R-CNN,
đánh giá bằng cùng một bộ mã và xem kết quả trong giao diện Streamlit.

## Demo trực tuyến

Bản Vercel tại https://pcbnhom4.vercel.app/ hỗ trợ tải **một ảnh để so sánh YOLOv8n và YOLO11n**. Có thể chọn từng vùng lỗi và xuất kết quả. Xem `vercel_app/README.md` để biết cấu hình, kiểm tra ONNX và kết quả YOLO11n mới.

Triển khai trên Streamlit Community Cloud với tệp `app.py` và Python 3.11.
Bản demo chạy mô hình YOLOv8n trên CPU. Chọn ảnh mẫu DeepPCB để xem dự đoán
ngay hoặc chuyển sang **Tải ảnh của bạn**. Bản demo có năm ảnh mẫu và kết quả
đánh giá đã lưu của năm lượt thực nghiệm. Để chuẩn bị dữ liệu và huấn luyện lại,
cần tải bộ DeepPCB đầy đủ theo hướng dẫn bên dưới.

Ảnh mẫu lấy từ [DeepPCB](https://github.com/tangsanli5201/DeepPCB).
Giấy phép nguồn nằm tại `data/raw/DeepPCB/LICENSE`.

## Mở chương trình trên máy này

Nhấp đúp **MO_GIAO_DIEN.cmd** trong File Explorer. Tệp này tự khởi động dịch vụ ở nền,
kiểm tra sẵn sàng rồi mở trình duyệt. Cửa sổ mở chương trình có thể đóng sau khi thành công.
Sau khi khởi động lại máy, chạy lại tệp này trước khi truy cập trang.
Địa chỉ mặc định là http://127.0.0.1:8501; nếu cổng bận, chọn cổng trống và lưu địa chỉ thật trong
`outputs/server.json`. Chạy lại tệp mở sẽ sử dụng dịch vụ đang hoạt động, tránh tạo bản trùng.

**Không nhầm checkpoint kiểm tra kỹ thuật với mô hình đã huấn luyện đầy đủ.**
Các lần chạy có `--limit-train` hoặc `--limit-val` được gắn `smoke_test=true` và giao diện cảnh báo.
Một mô hình smoke test có thể chưa phát hiện được lỗi nào ở ngưỡng thông thường.
Không dùng các chỉ số đó làm kết quả của Chương 3.

## Cấu trúc

```text
pcb_project/
  app.py                    Giao diện phát hiện, ảnh hiệu, bảng kết quả và dữ liệu
  pcb/data.py               Kiểm tra cặp ảnh, nhãn và chia dữ liệu
  pcb/engine.py             Mạng, bộ nạp dữ liệu, huấn luyện, suy luận
  pcb/metrics.py            Đánh giá chung cho tất cả mô hình
  pcb/__main__.py           Các lệnh prepare/train/evaluate/predict
  scripts/                  Tải dữ liệu và cài môi trường
  tests/test_core.py        Kiểm tra nhãn, ảnh hiệu, ghép hộp và AP
  data/raw/DeepPCB/         Bộ dữ liệu chính thức
  data/processed/           Manifest, chia tập, báo cáo nhãn và ảnh kiểm tra
  outputs/<lần chạy>/       Cấu hình, trọng số, lịch sử, kết quả đánh giá
  .venv/                    Python và thư viện riêng của project
```

## 1. Môi trường

Project này có môi trường riêng `.venv`; không cần kích hoạt môi trường hoặc sửa execution policy.
Trong PowerShell, đi tới thư mục project:

```powershell
cd "<đường dẫn thư mục chứa app.py>"
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Nếu mang project sang máy khác, cài Python 3.11/3.12 rồi tạo môi trường:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Máy đang dùng được thiết lập bản PyTorch CPU để kiểm thử. Nếu có NVIDIA GPU và muốn huấn luyện
nhanh hơn, chọn bộ lệnh phù hợp tại https://pytorch.org/get-started/locally/ rồi cài vào chính `.venv`.
Không mặc định rằng `CUDA: False` có nghĩa phần cứng không có GPU.
Phiên bản đã cài được lưu trong `requirements-lock.txt`; mỗi lần huấn luyện cũng ghi `environment.json`.

## 2. Dữ liệu và nhãn

Kho gốc: https://github.com/tangsanli5201/DeepPCB (giữ LICENSE và thông tin nguồn).
Nếu chưa tải dữ liệu, chạy `scripts/download_data.ps1` hoặc tải ZIP từ kho chính thức.
Sau đó chạy **CHUAN_BI_DU_LIEU.cmd**, hoặc:

```powershell
.\.venv\Scripts\python.exe -m pcb prepare
```

Sản phẩm: `data/processed/manifest.json`, `manifest.csv`, `splits.json`, `class_mapping.json`, `audit.json`
và các ảnh nhãn minh họa trong `previews`. Các nhãn YOLO chuẩn hóa được xuất trong `yolo_labels/<tập>/`;
vòng huấn luyện của project đọc manifest và chuẩn hóa hộp khi tạo từng lô.

- Giữ 500 cặp Test theo danh sách gốc; tách 800 Train / 200 Validation từ 1000 cặp Train gốc, seed 42.
- Chia theo cặp ảnh; **không tuyên bố độc lập theo thiết kế mạch**. `group` nguồn được ghi trong manifest.
- Kiểm tra ảnh Test trùng nội dung giữa các tập; gặp trùng thì dừng để rà soát, không âm thầm bỏ qua.
- Nhãn gốc: 1 open, 2 short, 3 mousebite, 4 spur, **5 copper**, **6 pin-hole**.
- Mã chuẩn nội bộ/YOLO: 0–5; Faster R-CNN dùng 1–6 và dành 0 cho nền.
- Tọa độ nguồn được giữ theo xyxy và dùng như cạnh liên tục; không tự cộng/trừ 1 pixel.
  Kiểm tra ảnh nhãn trong `previews` trước khi bắt đầu huấn luyện dài.
- Ảnh gốc và nhãn gốc không bị ghi đè. Đầu vào hai kênh được tạo trực tiếp từ cặp ảnh, không lưu vào JPEG.

## 3. Chạy kiểm tra kỹ thuật trước

Lệnh sau chạy trên dữ liệu thật nhưng chỉ dùng tập rất nhỏ. Mục đích là xác minh luồng chương trình:

```powershell
.\.venv\Scripts\python.exe -m pcb train --model yolov8n --mode test --name smoke_yolov8 --epochs 1 --imgsz 160 --batch 2 --limit-train 8 --limit-val 4 --scratch --device cpu
```

`--scratch` không tải trọng số học trước. Các lần chạy có tên riêng và sẽ từ chối ghi đè thư mục cũ;
hãy đổi `--name` nếu thử lại. Checkpoint là định dạng riêng của project, không truyền trực tiếp vào
CLI `yolo` và không chọn trọng số COCO gốc trong giao diện.

## 4. Huấn luyện thực nghiệm đầy đủ

Nhấp **HUAN_LUYEN_YOLO.cmd** hoặc chạy:

```powershell
.\.venv\Scripts\python.exe -m pcb train --model yolov8n --mode test --name yolov8n_test --epochs 100 --imgsz 640 --batch 8
.\.venv\Scripts\python.exe -m pcb train --model yolo11n --mode test --name yolo11n_test --epochs 100 --imgsz 640 --batch 8
```

Mặc định dùng trọng số học trước (tự tải lần đầu), AdamW, lr 0.001, weight decay 0.0005, dừng sớm
sau 20 epoch không cải thiện mAP50-95 trên Validation, seed 42. Chương 2 có cấu hình khảo sát SGD;
**mã thực thi ở đây dùng AdamW**: ghi đúng cấu hình thực tế khi hoàn thiện báo cáo.

Chọn họ YOLO dựa trên Validation rồi chạy cùng họ cho hai dạng đầu vào khác. Ví dụ chọn YOLOv8n:

```powershell
.\.venv\Scripts\python.exe -m pcb train --model yolov8n --mode diff --name yolov8n_diff --epochs 100 --imgsz 640 --batch 8
.\.venv\Scripts\python.exe -m pcb train --model yolov8n --mode pair --name yolov8n_pair --epochs 100 --imgsz 640 --batch 8
.\.venv\Scripts\python.exe -m pcb train --model fasterrcnn --mode test --name fasterrcnn_test --epochs 100 --imgsz 640 --batch 2 --lr 0.0001
```

`test` nghĩa là **ảnh kiểm tra làm đầu vào**, không phải dùng tập Test để huấn luyện.
`diff` là ảnh hiệu tuyệt đối, lặp thành ba kênh. `pair` là [ảnh mẫu, ảnh kiểm tra] hai kênh.
Mô hình hai kênh thay lớp nhận đầu vào và khởi tạo mỗi kênh bằng nửa tổng trọng số RGB của lớp cũ.

Project dùng vòng huấn luyện PyTorch riêng để cả ba dạng đầu vào được xử lý nhất quán. Không dùng
ngầm bộ nạp RGB mặc định của Ultralytics cho tensor hai kênh. Loss YOLO lấy từ DetectionModel;
YOLO được suy luận bằng decoder/NMS của Ultralytics. Faster R-CNN lấy từ TorchVision.
Pipeline này không bao gồm toàn bộ tiện ích trainer chính thức (EMA, Mosaic, lịch learning rate tự động).

Nếu hết bộ nhớ, giảm `--batch`. Khi dùng GPU được cài đúng, `--device auto` tự chọn CUDA;
có thể chỉ định `--device cuda:0`. Huấn luyện 100 epoch trên CPU có thể kéo dài; lệnh này không tự chạy
khi mở giao diện. `Ctrl+C` dừng và giữ các checkpoint đã lưu trước đó.

Mỗi lần chạy lưu:
- `config.json`, `environment.json`, `sample_ids.json`: cấu hình và mẫu dùng thực tế.
- `history.csv`, `learning_curves.png`: quá trình học.
- `weights/best.pt`, `weights/last.pt`: chọn best theo mAP50-95 Validation, giữ last.
- `validation.json`, `status.json`: chỉ số và trạng thái chạy.

## 5. Đánh giá và nội dung Chương 3

Sau khi chọn cấu hình bằng Validation, đánh giá lại Train và đánh giá Test:

```powershell
.\.venv\Scripts\python.exe -m pcb evaluate --checkpoint outputs/yolov8n_test/weights/best.pt --split train
.\.venv\Scripts\python.exe -m pcb evaluate --checkpoint outputs/yolov8n_test/weights/best.pt --split test
```

Lặp lại với các checkpoint khác. Mỗi lần đánh giá xuất `metrics.json`, `per_class.csv`,
`predictions.json` và một số ảnh minh họa. Giao diện đọc được các báo cáo này.

Quy tắc chung: ghép một-một, đúng lớp, sắp theo điểm, AP nội suy 101 mức Recall;
mAP50 và trung bình IoU 0.50:0.05:0.95. Các lớp không có nhãn trong tập con không được tính vào
trung bình AP; full Test có đủ lớp. Precision/Recall/F1 dùng ngưỡng điểm 0.25 và IoU 0.5 mặc định.
AP nhận dự đoán ở ngưỡng 0.001, tách biệt với ngưỡng hiển thị. Đây là bộ đánh giá chung của project,
**không phải bộ DeepPCB gốc với ngưỡng IoU > 0.33**, không tuyên bố đồng nhất tuyệt đối với COCO API.

Thời gian báo cáo gồm đổi kích thước + tạo tensor + mạng + hậu xử lý; loại làm nóng và không tính
đọc/lưu file. Không ghi số này thành thời gian toàn bộ ứng dụng.

## 6. Mở giao diện và dự đoán

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Giao diện tự phát hiện lỗi khi bạn tải một ảnh PCB hoặc chọn ảnh kiểm tra từ DeepPCB.
Kết quả tự cập nhật khi đổi ảnh, mô hình hoặc ngưỡng điểm tin cậy.
Với ảnh tải lên có độ tương phản thấp, có thể bật **Tăng tương phản ảnh** để thử lại.
Khi cạnh dài hơn hai lần kích thước đầu vào của mô hình, giao diện quét các vùng chồng lấn
ở độ phân giải gốc, đổi tọa độ về ảnh toàn bo và gộp các khung trùng cùng loại.
Tab **Phát hiện** chỉ liệt kê các checkpoint huấn luyện với đầu vào là ảnh kiểm tra.
Ảnh đầu vào và ảnh kết quả hiển thị cạnh nhau. Chọn một dòng trong bảng vùng lỗi để
khoanh tròn vị trí dự đoán trên ảnh kết quả. Bảng vùng lỗi và tệp PNG/JSON nằm trong
**Chi tiết dự đoán và tải kết quả**. Tab **Kết quả thực nghiệm** đọc cả các phương án ảnh hiệu và hai kênh.

Để dự đoán bằng checkpoint dùng cặp ảnh, chạy dòng lệnh:

```powershell
.\.venv\Scripts\python.exe -m pcb predict --checkpoint outputs/yolov8n_pair_full_01/weights/best.pt --template "duong_dan_anh_mau.png" --tested "duong_dan_anh_kiem_tra.png"
```

Ảnh phải cùng vùng mạch và đã căn chỉnh. Project kiểm tra kích thước, không tự cam kết kiểm tra
đúng thiết kế chỉ từ tên tệp. Không có dự đoán không có nghĩa mạch chắc chắn không lỗi.

## 7. Kiểm thử mã

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Đọc `VALIDATION.md` để biết chính xác những gì đã chạy trong phiên tạo project.
Đọc tài liệu báo cáo ở thư mục cha để hoàn thiện Chương 3 bằng số liệu thật của các lần chạy đầy đủ.

## Bắt đầu sử dụng sau khi nhận project

1. Nhấp `MO_GIAO_DIEN.cmd` để xem ngay ảnh dữ liệu và thử checkpoint minh họa.
2. Xem `data/processed/previews` để kiểm tra nhãn.
3. Khi sẵn sàng dành thời gian huấn luyện, chạy `HUAN_LUYEN_YOLO.cmd`.
4. Sau khi huấn luyện đầy đủ và chọn cấu hình, chạy `DANH_GIA_YOLO.cmd` để xuất bảng Train/Test.
5. Dùng lệnh trong mục 4 để bổ sung các thí nghiệm ảnh hiệu, hai kênh và Faster R-CNN.
