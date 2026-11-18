# PCB Lab trên Vercel

Ứng dụng phát hiện sáu loại lỗi PCB từ một ảnh, gồm giao diện web và API FastAPI.
Mô hình YOLOv8n đã huấn luyện được xuất sang ONNX và suy luận bằng ONNX Runtime CPU.

Trong Vercel, nhập kho GitHub `aitongvan/pcb-lab-demo`, chọn Root Directory
`vercel_app` và Framework Preset `FastAPI`. Phiên bản Python được khai báo trong
`.python-version`. Cài đặt thư viện từ `requirements.txt`.

Chạy trên máy bằng `uvicorn app:app --host 127.0.0.1 --port 8502` trong thư mục này.

API `/api/predict` nhận ảnh, ngưỡng tin cậy và tùy chọn tăng tương phản. API
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
