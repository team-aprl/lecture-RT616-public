param([switch]$Cpu, [string]$Python)
$ErrorActionPreference = 'Stop'
$setupArgs = @((Join-Path $PSScriptRoot 'setup.py'))
if ($Cpu) { $setupArgs += '--cpu' }
if ($Python) { & $Python @setupArgs } else { & py -3.12 @setupArgs }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
