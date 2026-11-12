@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
echo Huấn luyện 100 epoch trên dữ liệu Train, chọn mô hình bằng Validation.
echo Có thể mất nhiều thời gian trên CPU. Ctrl+C để dừng.
".venv\Scripts\python.exe" -m pcb train --model yolov8n --mode test --name yolov8n_test --epochs 100 --imgsz 640 --batch 8
pause

