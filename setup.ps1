# Install pretrained recognition; this script never trains YOLO.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create Python environment. Install Python with Tk support.' }
}
& .\.venv\Scripts\python.exe -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements-collectorvision.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe recognition.py --prepare
if ($LASTEXITCODE -ne 0) { throw 'Catalog/model preparation failed.' }
Write-Host 'Ready. Start the scanner with .\run.ps1'
