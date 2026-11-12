param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$pcbRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $pcbRoot
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Khong tao duoc venv. Hay cai Python 3.11/3.12 hoac truyen -Python duong_dan_python.exe.' }
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cai thu vien that bai.' }
& '.\.venv\Scripts\python.exe' -m pip freeze | Set-Content -Encoding utf8 requirements-lock.txt
Write-Host 'Da cai xong. Chay CHUAN_BI_DU_LIEU.cmd va MO_GIAO_DIEN.cmd.'
