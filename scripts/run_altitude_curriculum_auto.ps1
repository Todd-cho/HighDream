param(
    [string]$Python = "C:\Users\2bpro\anaconda3\envs\aip\python.exe",
    [ValidateRange(0, 20)]
    [int]$MaxRestarts = 3,
    [ValidateRange(1, 200)]
    [int]$SummaryWindow = 20
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FreshYaml = Join-Path $ProjectRoot "experiments\curriculum_altitude_sac.yaml"
$ResumeYaml = Join-Path $ProjectRoot "experiments\curriculum_altitude_sac_resume.yaml"
$RunDir = Join-Path $ProjectRoot "artifacts\curriculum\highdream\altitude_curriculum_v1"
$StateFile = Join-Path $RunDir "curriculum_state.json"
$SummaryScript = Join-Path $PSScriptRoot "summarize_curriculum.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python not found: $Python"
}
foreach ($RequiredFile in @($FreshYaml, $ResumeYaml, $SummaryScript)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}

Set-Location -LiteralPath $ProjectRoot

# Read-only preflight: imports reward/curriculum and validates every stage.
& $Python -B -c "from student.my_reward import MY_REWARD_CONFIG; from student.my_curriculum import get_stages; from dogfight.envs.initial_scenario import validate_scenario_pool; stages=get_stages(); [validate_scenario_pool(s.env_overrides['initial_scenario']) for s in stages if s.env_overrides.get('initial_scenario',{}).get('mode')=='scenario_pool']; assert all(s.randomization.get('enabled') is False for s in stages if s.env_overrides.get('initial_scenario',{}).get('mode')=='scenario_pool'); print('[preflight] reward, stages, and scenario pools: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Preflight validation failed. Training was not started."
}

$Attempt = 0
while ($true) {
    $UseResume = Test-Path -LiteralPath $StateFile -PathType Leaf
    $Yaml = if ($UseResume) { $ResumeYaml } else { $FreshYaml }
    $Mode = if ($UseResume) { "resume" } else { "fresh" }
    Write-Host "[auto] attempt=$($Attempt + 1) mode=$Mode yaml=$Yaml"

    & $Python "scripts\run_experiment.py" $Yaml
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        break
    }
    if ($Attempt -ge $MaxRestarts) {
        throw "Training failed with exit code $ExitCode after $($Attempt + 1) attempts. Check emergency checkpoint under $RunDir."
    }

    $Attempt += 1
    Write-Warning "Training exited with code $ExitCode. Resuming from the saved curriculum state (retry $Attempt/$MaxRestarts)."
    Start-Sleep -Seconds 5
}

if (Test-Path -LiteralPath $StateFile -PathType Leaf) {
    & $Python -B $SummaryScript $RunDir --window $SummaryWindow
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Training ended, but result summarization failed. Raw files remain in $RunDir."
    }
}

Write-Host "[auto] finished. Results: $RunDir"
