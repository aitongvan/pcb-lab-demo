@echo off
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua co moi truong Python. Xem README.md de cai dat.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" scripts\launch.py
if errorlevel 1 (
  echo.
  echo Khong mo duoc ung dung. Hay gui thong bao loi trong cua so nay.
  pause
)

