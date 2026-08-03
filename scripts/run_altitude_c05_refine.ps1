param(
    [string]$Python = "C:\Users\2bpro\anaconda3\envs\aip\python.exe",
    [ValidateRange(20, 500)] [int]$ScreenIterations = 100,
    [ValidateRange(50, 1000)] [int]$ValidationIterations = 150,
    [ValidateRange(1, 8)] [int]$TopK = 3,
    [ValidateRange(1, 10)] [int]$Repeats = 3,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Runner = Join-Path $PSScriptRoot "run_altitude_c05_refine.py"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) { throw "Runner not found: $Runner" }
Set-Location -LiteralPath $ProjectRoot

$Arguments = @(
    "-B", $Runner,
    "--screen-iterations", $ScreenIterations,
    "--validation-iterations", $ValidationIterations,
    "--top-k", $TopK,
    "--repeats", $Repeats
)
if ($Resume) { $Arguments += "--resume" }
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "C05 refinement failed with exit code $LASTEXITCODE" }

Write-Host "[done] artifacts\altitude_c05_refine_v1\validation_summary.csv"
