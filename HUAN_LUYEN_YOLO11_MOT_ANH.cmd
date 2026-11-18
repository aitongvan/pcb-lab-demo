@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PCB_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%PCB_PYTHON%" (
  echo Khong tim thay .venv. Hay dat file nay vao thu muc project co .venv va thu muc pcb.
  goto failed
)
"%PCB_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'Can PyTorch CUDA va GPU NVIDIA'; print('GPU:', torch.cuda.get_device_name(0))"
if errorlevel 1 goto failed
"%PCB_PYTHON%" -c "from pcb.data import manifest; n=[len(manifest(s)) for s in ['train','val','test']]; print('Train/Val/Test:',n); assert n==[800,200,500], 'Can dung bo chia du lieu goc'"
if errorlevel 1 goto failed
if exist "outputs\yolo11n_single_full_01" (
  echo Thu muc yolo11n_single_full_01 da ton tai. Khong ghi de. Xem huong dan truoc khi chay lai.
  goto failed
)
"%PCB_PYTHON%" -m pcb train --model yolo11n --mode test --name yolo11n_single_full_01 --epochs 100 --imgsz 640 --batch 2 --lr 0.001 --patience 20 --seed 42 --confidence 0.25 --device cuda:0
if errorlevel 1 goto failed
call "%~dp0XUAT_YOLO11_MOT_ANH.cmd"
exit /b %errorlevel%
:failed
echo Chua hoan thanh. Giu nguyen thong bao loi de kiem tra.
pause
exit /b 1
