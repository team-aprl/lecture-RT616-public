param([switch]$Cpu, [int]$Port = 8766, [string]$Python)
$ErrorActionPreference = 'Stop'
$runArgs = @((Join-Path $PSScriptRoot 'run.py'), '--port', $Port)
if ($Cpu) { $runArgs += '--cpu' }
if ($Python) { & $Python @runArgs } else { & py -3.12 @runArgs }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }