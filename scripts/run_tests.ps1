<#
.SYNOPSIS
  Run the camdetect, Causeway, and eval test suites (and repo doc checks).

.DESCRIPTION
  Uses .venv\Scripts\python.exe at the repo root when present, otherwise
  `py -3.11`. Create the venv once:

    py -3.11 -m venv .venv
    .venv\Scripts\pip install -r camdetect/requirements-dev.txt -r Causeway/requirements-dev.txt -r eval/requirements-dev.txt
    # slow tests (TensorFlow):
    .venv\Scripts\pip install -r eval/requirements-notebook.txt

.PARAMETER Slow
  Also run tests marked `slow` (TensorFlow, longer model training).

.PARAMETER Coverage
  Print a coverage report per suite.

.PARAMETER Suite
  Limit to one or more of: camdetect, Causeway, eval, repo.

.EXAMPLE
  scripts\run_tests.ps1
  scripts\run_tests.ps1 -Slow -Coverage
  scripts\run_tests.ps1 -Suite eval
#>
param(
    [switch]$Slow,
    [switch]$Coverage,
    [string[]]$Suite = @('camdetect', 'Causeway', 'eval', 'repo')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if (Test-Path $venvPy) { $py = @($venvPy) } else { $py = @('py', '-3.11') }

# `powershell -File` passes "a,b" as one string; accept both forms.
$Suite = @($Suite | ForEach-Object { $_ -split ',' } | Where-Object { $_ })

$failed = @()
foreach ($name in $Suite) {
    $dir = if ($name -eq 'repo') { Join-Path $root 'scripts' } else { Join-Path $root $name }
    if (-not (Test-Path (Join-Path $dir 'tests'))) { Write-Host "skip $name (no tests/)"; continue }
    $pytestArgs = @('-m', 'pytest', '-q', '-p', 'no:cacheprovider')
    if ($Slow) { $pytestArgs += @('-m', 'slow or not slow') }
    if ($Coverage -and $name -ne 'repo') { $pytestArgs += @('--cov=.', '--cov-report=term-missing:skip-covered') }
    Write-Host "=== $name" -ForegroundColor Cyan
    Push-Location $dir
    try {
        $exe = $py[0]
        $rest = @($py | Select-Object -Skip 1) + $pytestArgs
        & $exe @rest
        if ($LASTEXITCODE -ne 0) { $failed += $name }
    } finally {
        Pop-Location
    }
}

if ($failed.Count -gt 0) {
    Write-Host "FAILED: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host 'All suites passed.' -ForegroundColor Green
