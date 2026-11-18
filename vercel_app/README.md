# PCB Lab trên Vercel

Ứng dụng phát hiện sáu loại lỗi PCB từ một ảnh, gồm giao diện web và API FastAPI.
Hai mô hình YOLOv8n và YOLO11n dùng một ảnh đã được huấn luyện, xuất sang ONNX và suy luận bằng ONNX Runtime CPU. Web công khai: https://pcbnhom4.vercel.app/.

Trong Vercel, nhập kho GitHub `aitongvan/pcb-lab-demo`, chọn Root Directory
`vercel_app` và Framework Preset `FastAPI`. Phiên bản Python được khai báo trong
`.python-version`. Cài đặt thư viện từ `requirements.txt`.

Chạy trên máy bằng `uvicorn app:app --host 127.0.0.1 --port 8502` trong thư mục này.

API `/api/predict` nhận ảnh và tùy chọn tăng tương phản, mặc định ngưỡng 0,05. Giao diện cố định ngưỡng này và ẩn thanh điều chỉnh. `/api/models` cung cấp trạng thái mô hình; `/api/compare` nhận cùng ảnh và chạy cả hai mô hình với cùng ngưỡng, tiền xử lý và NMS. Mỗi mô hình được làm nóng trước khi đo thời gian. API
`/api/export` xuất PNG hoặc GIF quét một lượt dựa trên kết quả dự đoán.
Giới hạn ảnh đầu vào là 4 MB và 24 triệu điểm ảnh. Ảnh lớn được quét theo các
vùng chồng lấn giống phiên bản gốc.

`scripts/export_model.py` xuất checkpoint gốc. `scripts/verify_model.py` đối chiếu
13 tình huống giữa ONNX và PyTorch, gồm ảnh mẫu, ảnh chữ nhật, tăng tương phản và
quét theo vùng. `scripts/verify_api.py` kiểm tra nhận diện, xuất GIF một lượt,
xuất PNG và báo lỗi đầu vào. Các công cụ xuất và đối chiếu cần thư viện huấn luyện
ở môi trường phát triển, ngoài các thư viện chạy web trong `requirements.txt`.

Ảnh mẫu lấy từ [DeepPCB](https://github.com/tangsanli5201/DeepPCB).
Giấy phép nguồn nằm ở thư mục `data/raw/DeepPCB/LICENSE` của kho chính.

## YOLO11n một ảnh đã hoàn tất

Lượt `yolo11n_single_colab_02` huấn luyện trên Tesla T4, tối đa 100 epoch, dừng sớm ở epoch 55 và chọn checkpoint epoch 35 bằng Validation. Dùng đúng ID Train/Validation của YOLOv8n: 800/200 cặp; Test chính thức 500 cặp. Batch 2, ảnh 640, AdamW, lr 0.001, seed 42, patience 20. Báo cáo có nhãn dùng ngưỡng 0,25, IoU ghép nhãn 0,5.

| Tập | mAP@50 | mAP@50–95 |
|---|---:|---:|
| Train | 98,51% | 79,71% |
| Validation | 96,21% | 75,24% |
| Test | 94,66% | 74,33% |

Các lượt gốc dùng RTX 3050 và Torch 2.7.1; lượt mới dùng Colab T4 và Torch 2.14.1. Giữ nguyên cách chia, nhãn, vòng lặp và siêu tham số chính nhưng môi trường không hoàn toàn đồng nhất; không coi chênh lệch là do duy nhất kiến trúc. Thời gian trên web đo riêng bằng ONNX Runtime CPU. Bảng thực nghiệm lưu chỉ số trên tập có nhãn; so sánh một ảnh không có nhãn chỉ đối chiếu dự đoán, không tính độ chính xác.
