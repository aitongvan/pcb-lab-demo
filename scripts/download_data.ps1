$ErrorActionPreference = 'Stop'
$pcbRoot = Split-Path -Parent $PSScriptRoot
$pcbTarget = Join-Path $pcbRoot 'data\raw\DeepPCB'
if (Test-Path -LiteralPath (Join-Path $pcbTarget 'PCBData\trainval.txt')) {
    Write-Host 'DeepPCB da ton tai.'
} else {
    git clone --depth 1 https://github.com/tangsanli5201/DeepPCB.git $pcbTarget
    if ($LASTEXITCODE -ne 0) { throw 'Khong tai duoc DeepPCB. Xem ket noi mang hoac tai ZIP tu GitHub.' }
}

