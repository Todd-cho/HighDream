param(
    [string]$Python = "C:\Users\2bpro\anaconda3\envs\aip\python.exe",
    [string]$Campaign = "altitude_reward_search_v2",
    [ValidateRange(2, 100)] [int]$ScreenTrials = 16,
    [ValidateRange(5, 500)] [int]$ScreenIterations = 30,
    [ValidateRange(1, 20)] [int]$TopK = 5,
    [ValidateRange(1, 10)] [int]$ValidationRepeats = 3,
    [ValidateRange(10, 1000)] [int]$ValidationIterations = 50,
    [int]$CandidateSeed = 260721,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SearchScript = Join-Path $PSScriptRoot "run_altitude_reward_search.py"
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python not found: $Python"
}
if (-not (Test-Path -LiteralPath $SearchScript -PathType Leaf)) {
    throw "Search script not found: $SearchScript"
}

Set-Location -LiteralPath $ProjectRoot
$ResumeArg = @()
if ($Resume) {
    $ResumeArg = @("--resume")
}

Write-Host "[campaign] Phase 1/2: $ScreenTrials candidates x $ScreenIterations iterations"
& $Python -B $SearchScript `
    --phase screen `
    --campaign $Campaign `
    --trials $ScreenTrials `
    --iterations $ScreenIterations `
    --seed $CandidateSeed `
    @ResumeArg
if ($LASTEXITCODE -ne 0) {
    throw "Screen phase failed with exit code $LASTEXITCODE"
}

Write-Host "[campaign] Phase 2/2: top $TopK x $ValidationRepeats repeats x $ValidationIterations iterations"
& $Python -B $SearchScript `
    --phase validate `
    --campaign $Campaign `
    --top-k $TopK `
    --repeats $ValidationRepeats `
    --iterations $ValidationIterations `
    @ResumeArg
if ($LASTEXITCODE -ne 0) {
    throw "Validation phase failed with exit code $LASTEXITCODE"
}

$ResultRoot = Join-Path $ProjectRoot "artifacts\$Campaign"
Write-Host "[campaign] Complete"
Write-Host "[campaign] Ranked candidates: $(Join-Path $ResultRoot 'validation_summary.csv')"
