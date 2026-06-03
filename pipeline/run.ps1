# Run detection pipeline for all cameras
# Usage: .\pipeline\run.ps1
# Or for a single camera: .\pipeline\run.ps1 -Cam 1

param(
    [int]$Cam = 0,
    [int]$MaxFrames = 0
)

$ErrorActionPreference = "Stop"

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Purplle Store Intelligence - Detection Pipeline" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

$pythonExe = "python"
$detectScript = "$PSScriptRoot\detect.py"

if ($Cam -gt 0) {
    Write-Host "Processing Camera $Cam only..." -ForegroundColor Yellow
    if ($MaxFrames -gt 0) {
        & $pythonExe $detectScript --cam $Cam --max-frames $MaxFrames
    } else {
        & $pythonExe $detectScript --cam $Cam
    }
} else {
    Write-Host "Processing ALL 5 cameras..." -ForegroundColor Yellow
    if ($MaxFrames -gt 0) {
        & $pythonExe $detectScript --all --max-frames $MaxFrames
    } else {
        & $pythonExe $detectScript --all
    }
}

Write-Host ""
Write-Host "Pipeline complete! Events in data/events/" -ForegroundColor Green
Write-Host "Next: POST events to API with:" -ForegroundColor Green
Write-Host "  python pipeline/ingest_to_api.py" -ForegroundColor White
