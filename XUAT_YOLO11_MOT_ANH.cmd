@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PCB_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "outputs\yolo11n_single_full_01\weights\best.pt" goto failed
"%PCB_PYTHON%" -c "import json; s=json.load(open('outputs/yolo11n_single_full_01/status.json')); assert s['status']=='completed' and not s['smoke_test'], 'Training chua hoan thanh'"
if errorlevel 1 goto failed
for %%S in (val train test) do (
  "%PCB_PYTHON%" -m pcb evaluate --checkpoint outputs/yolo11n_single_full_01/weights/best.pt --split %%S --confidence 0.25 --device cuda:0
  if errorlevel 1 goto failed
)
"%PCB_PYTHON%" -m pip install "onnx>=1.16,<2" "onnxruntime>=1.19,<2"
if errorlevel 1 goto failed
"%PCB_PYTHON%" scripts/export_yolo11_single.py
if errorlevel 1 goto failed
echo HOAN THANH. Gui lai toan bo outputs\yolo11n_single_full_01 de ghep vao web.
pause
exit /b 0
:failed
echo Chua xuat duoc. Kiem tra best.pt, status.json va thong bao loi o tren.
pause
exit /b 1
