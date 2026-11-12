@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist "outputs\yolov8n_test\weights\best.pt" (
  echo Chua co mo hinh day du. Chay HUAN_LUYEN_YOLO.cmd truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pcb evaluate --checkpoint outputs/yolov8n_test/weights/best.pt --split train
if errorlevel 1 goto end
".venv\Scripts\python.exe" -m pcb evaluate --checkpoint outputs/yolov8n_test/weights/best.pt --split test
:end
pause
