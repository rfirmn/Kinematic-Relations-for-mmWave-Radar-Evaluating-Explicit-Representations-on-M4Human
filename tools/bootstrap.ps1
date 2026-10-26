param([Parameter(Mandatory=$true)][string]$Python)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed' }
}
& '.\.venv\Scripts\python.exe' -m pip install uv
if ($LASTEXITCODE -ne 0) { throw 'uv installation failed' }
$taskLock = 'docs/report_training/fresh_20261026_01/requirements.windows.lock.txt'
& '.\.venv\Scripts\uv.exe' --cache-dir '.cache/uv' pip install --python '.venv\Scripts\python.exe' -r $taskLock
if ($LASTEXITCODE -ne 0) { throw 'Locked dependency installation failed' }
& '.\.venv\Scripts\uv.exe' --cache-dir '.cache/uv' pip install --python '.venv\Scripts\python.exe' -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
& '.\.venv\Scripts\python.exe' -m relational64.cli verify --suite all
exit $LASTEXITCODE
