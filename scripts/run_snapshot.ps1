# Runs the FPL snapshot ingest and appends output to a rolling log.
# Invoked by the "FPL Snapshot" scheduled task (see scripts/register_snapshot_task.ps1).

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$logDir = Join-Path $root "data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "snapshot.log"

Set-Location $root
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "----- $timestamp -----"
& $python -m ingest.snapshot --season 2026-27 *>> $logFile
