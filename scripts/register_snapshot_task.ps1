# Registers/updates the "FPL Snapshot" Windows Scheduled Task: three runs a week
# (Friday evening, Saturday pre-deadline, Tuesday morning), matching the cadence
# in fpl/ingest/snapshot.py and README.md.
#
# Re-run this script any time to update times; it replaces the existing task.

$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $root "scripts\run_snapshot.ps1"
$taskName = "FPL Snapshot"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

$triggers = @(
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday   -At 18:00
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At 10:00
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday  -At 09:00
)

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $triggers -Settings $settings `
    -Description "Snapshots the FPL API (bootstrap-static, fixtures, live) for backtest ground truth. Friday evening, Saturday pre-deadline, Tuesday post-gameweek." `
    -Force | Out-Null

Write-Host "Registered task '$taskName' with $($triggers.Count) triggers."
Get-ScheduledTask -TaskName $taskName | Get-ScheduledTaskInfo | Format-List TaskName, LastTaskResult, NextRunTime
