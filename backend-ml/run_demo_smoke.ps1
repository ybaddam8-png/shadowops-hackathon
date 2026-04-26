$ErrorActionPreference = "Stop"

$pythonCmd = ".\.venv311\Scripts\python"
if (!(Test-Path $pythonCmd)) {
    $pythonCmd = "python"
}

Write-Host "Starting ShadowOps API smoke server on :8012..."
$proc = Start-Process -FilePath $pythonCmd -ArgumentList @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8012") -PassThru -WorkingDirectory (Get-Location)

try {
    Start-Sleep -Seconds 3
    $env:SHADOWOPS_BASE_URL = "http://127.0.0.1:8012"
    & $pythonCmd test_api.py
    & $pythonCmd test_all.py
}
finally {
    if ($proc -and !$proc.HasExited) {
        Stop-Process -Id $proc.Id -Force
    }
}
