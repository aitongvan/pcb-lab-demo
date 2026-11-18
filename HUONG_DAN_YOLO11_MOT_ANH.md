# Huấn luyện YOLO11n một ảnh để so sánh với YOLOv8n

## Trên máy RTX 3050 6 GB

1. Giải nén gói bổ sung vào đúng thư mục project đang chạy được trên máy RTX. Thư mục đó phải có `.venv`, `pcb`, `data`, `outputs`. Giữ nguyên thư mục `scripts` trong gói, không đặt toàn bộ gói vào một thư mục con.
2. Cắm sạc, tắt tự ngủ. Đóng các chương trình đang dùng nhiều GPU.
3. Mở `HUAN_LUYEN_YOLO11_MOT_ANH.cmd`. Hoặc mở PowerShell trong project và chạy `./HUAN_LUYEN_YOLO11_MOT_ANH.cmd`.
4. Chờ xuất hiện thông báo `HOAN THANH`. Việc học có thể mất nhiều giờ; thời gian thực tế phụ thuộc máy. Chương trình in kết quả sau từng epoch.
5. Nén và gửi lại **toàn bộ thư mục `outputs/yolo11n_single_full_01`**. Cần giữ `weights/best.pt`, `config.json`, `status.json`, `history.csv`, `sample_ids.json`, `environment.json`, `validation.json`, các thư mục `evaluation_val`, `evaluation_train`, `evaluation_test` và `deploy`.

Không cần chép `.venv` từ máy hiện tại sang máy RTX. Dùng môi trường CUDA đã chạy thành công trước đây trên máy RTX.

## Bộ chạy thực hiện những gì

- Kiểm tra CUDA hoạt động và số lượng Train/Validation/Test là 800/200/500. Không tự chia lại dữ liệu.
- Huấn luyện YOLO11n với `--mode test`. Trong project, **test ở tham số mode có nghĩa là dùng một ảnh kiểm tra**, không có nghĩa là lấy tập Test để huấn luyện.
- Ảnh 640×640, batch 2, AdamW, learning rate 0.001, seed 42, tối đa 100 epoch, early stopping patience 20. Các thiết lập này khớp cấu hình đối chứng YOLOv8n đã có.
- Chọn best checkpoint bằng mAP@50–95 trên Validation. Sau khi học xong mới đánh giá Train, Validation và Test. Không điều chỉnh mô hình theo kết quả Test.
- Cài ONNX và ONNX Runtime phục vụ xuất mô hình; không cài lại Torch/CUDA.
- Xuất mô hình thật đã học sang `deploy/pcb_yolo11n.onnx`, đầu vào 1×3×640×640. Đối chiếu đầu ra ONNX với PyTorch trên 5 ảnh Validation, lưu thông tin vào `deploy/model_info.json`.

Để đối chiếu chặt chẽ, khi nhận kết quả cần kiểm tra ID của từng tập và phiên bản thư viện so với lượt YOLOv8n gốc. Cùng số lượng ảnh chưa đủ bảo đảm cùng tập dữ liệu. Nếu phiên bản khác đáng kể, nên chạy lại hai mô hình trong cùng môi trường và ghi rõ trong báo cáo.

## Nếu gặp lỗi

- Không có CUDA: kiểm tra đang dùng đúng `.venv` trên máy RTX; không chuyển sang chạy CPU để thay kết quả học đầy đủ.
- Thư mục lượt chạy đã tồn tại: chương trình dừng để tránh ghi đè. Không xóa trọng số cũ. Gửi thông báo để chọn tên lượt chạy khác hoặc xử lý lượt học dở.
- Đã học xong nhưng lỗi đánh giá/xuất ONNX: chạy `XUAT_YOLO11_MOT_ANH.cmd`; bước này không huấn luyện lại.
- Giữ nguyên thông báo lỗi và thư mục đã tạo để kiểm tra. Không đổi tên bản hai kênh thành bản một ảnh.

## Ghép vào web sau khi huấn luyện

Web đã có sẵn bộ nhận diện hai mô hình, API so sánh và bảng đối chiếu. Sau khi kiểm tra kết quả, chép `deploy/pcb_yolo11n.onnx` vào `vercel_app/model/pcb_yolo11n.onnx`, đưa thông tin thực nghiệm mới vào web, kiểm tra suy luận và cập nhật GitHub. Vercel tự triển khai; chế độ so sánh mở khi cả hai tệp mô hình có mặt.

Ảnh không có nhãn chỉ giúp đối chiếu vị trí, loại lỗi, số dự đoán và tốc độ. Không suy ra mô hình chính xác hơn từ số lượng lỗi hoặc điểm tin cậy trung bình. Kết luận chất lượng dựa trên Precision, Recall, F1 và mAP của tập đánh giá có nhãn.
