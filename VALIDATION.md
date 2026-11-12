# Kiểm chứng project ngày 28/10/2026

## Môi trường đã cài

- Python 3.12.14 trong `.venv` của project.
- PyTorch 2.15.0+cpu; TorchVision 0.29.2+cpu; Ultralytics 8.4.191; Streamlit 1.66.1.
- Máy được kiểm tra: Intel Core i7-8665U và Intel UHD Graphics 620. Không phát hiện NVIDIA GPU qua thông tin hệ thống.
- `pip check`: không có yêu cầu thư viện bị hỏng. Chi tiết phiên bản: `requirements-lock.txt`.

## Dữ liệu thật đã chuẩn bị

- Đã clone kho DeepPCB chính thức, commit `08e98c4db5922613fb97176eb3d6497d48260cb1`.
- Đọc đủ 1500 cặp ảnh và nhãn. Tất cả hộp vượt qua kiểm tra tọa độ và mã lớp.
- Train: 800 cặp / 5485 hộp; Validation: 200 cặp / 1388 hộp; Test: 500 cặp / 3140 hộp.
- Không có ảnh kiểm tra trùng nội dung giữa các tập theo SHA-256 trên dữ liệu điểm ảnh.
- Đã xuất nhãn chuẩn hóa YOLO trong `data/processed/yolo_labels`.
- Đã xem trực quan hai ảnh nhãn minh họa chứa đủ sáu loại lỗi trong `data/processed/previews`.
- Chia Validation theo cặp ngẫu nhiên; không tuyên bố độc lập theo thiết kế mạch.

## Kiểm thử phần mềm đã thực hiện

- 11 kiểm thử đơn vị: IoU, AP với dự đoán đúng, hộp lặp, sai lớp, ghép hộp thay thế, dự đoán rỗng,
  tách ngưỡng điểm với AP, mã lớp, hộp không hợp lệ, ảnh hiệu không tràn số, sai kích thước cặp,
  chuẩn hóa nhãn (một số tình huống được kiểm tra chung trong cùng hàm test).
- Cả bốn cấu hình YOLOv8n/test, YOLOv8n/pair, YOLO11n/diff và Faster R-CNN/test đã:
  nạp hai cặp Train thật, tính mất mát hữu hạn, lan truyền ngược và suy luận thành công.
- Checkpoint hai kênh đã lưu/nạp lại; đầu ra được kiểm tra tính nhất quán.
- Nạp trọng số học trước cho YOLOv8n hai kênh thành công: lớp đầu có hình dạng (16, 2, 3, 3).
- Kiểm thử Streamlit: trang tải không có ngoại lệ, bấm Phát hiện lỗi, xuất hai nút tải kết quả,
  chuyển chế độ ảnh hiệu và chuyển chế độ tải ảnh.
- CLI predict và evaluate đã chạy thành công với checkpoint của project.

Kết quả máy đọc được: `outputs/integration_checks.json`, `outputs/ui_checks.json`.

## Những lần huấn luyện đã chạy

| Lần chạy | Dữ liệu | Epoch | Mục đích |
|---|---|---:|---|
| smoke_yolov8 | 8 Train / 4 Validation | 1 | Kiểm tra vòng huấn luyện từ trọng số ngẫu nhiên |
| smoke_pair | 4 Train / 2 Validation | 1 | Kiểm tra huấn luyện đầu vào hai kênh |
| demo_yolov8_pretrained | 32 Train / 16 Validation | 3 | Kiểm tra học chuyển giao và tạo checkpoint minh họa |

Lần demo dùng ảnh 320×320, batch 4, CPU và AdamW lr 0.001. Mất mát giảm từ khoảng 31.12 xuống 22.27.
mAP50 trên 16 ảnh Validation của lần chạy ngắn khoảng 0.099; mAP50-95 khoảng 0.042.
Ở ngưỡng điểm mặc định 0.25, lần demo ngắn chưa có dự đoán đạt tiêu chí ghép đúng trên tập con này.

**Các mô hình trên đều là chạy thử, không đủ để kết luận hiệu quả đề tài.** Giao diện có cảnh báo tương ứng.
Không có số liệu nào được tạo giả hoặc lấy từ bài báo rồi nhận là kết quả của project.

## Phần thực nghiệm chưa chạy đầy đủ

- Chưa huấn luyện 100 epoch trên 800 cặp Train.
- Chưa thực hiện thí nghiệm đầy đủ cho cả năm phương án đã đề xuất.
- Chưa đánh giá mô hình trên 500 cặp Test; giữ Test cho sau khi chốt mô hình bằng Validation.
- Chưa xác nhận khả năng tổng quát sang ảnh thu từ dây chuyền thực tế hoặc thiết kế PCB khác.

Mã và các lệnh cho các công việc trên có trong README. Không tự khởi động huấn luyện 100 epoch
khi mở project vì máy hiện tại chỉ được cài backend CPU và quá trình có thể kéo dài.
