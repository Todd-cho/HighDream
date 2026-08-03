param(
    [string]$Python = "C:\Users\2bpro\anaconda3\envs\aip\python.exe",
    [ValidateRange(10, 1000)] [int]$Iterations = 100,
    [ValidateRange(1, 10)] [int]$Repeats = 3,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_altitude_c050711.py"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) { throw "Runner not found: $Runner" }
Set-Location -LiteralPath $ProjectRoot

$Arguments = @("-B", $Runner, "--iterations", $Iterations, "--repeats", $Repeats)
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "C05/C07/C11 confirmation failed with exit code $LASTEXITCODE" }

Write-Host "[done] artifacts\altitude_c050711_confirm_v1\confirmation_summary.csv"
