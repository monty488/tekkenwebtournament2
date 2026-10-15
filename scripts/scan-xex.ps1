# Runs ttt2-scan (image dump, imports, helper addresses) and XenonAnalyse (jump tables).
. "$PSScriptRoot\common.ps1"

$xex = Join-Path $GameDir 'files\default.xex'
$patched = Join-Path $GameDir 'update\default_patched.xex'
if (Test-Path $patched) { $xex = $patched; Write-Host 'Using title-updated XEX' }
if (-not (Test-Path $xex)) { throw "Missing $xex. Run scripts\extract-game.ps1 first." }

Invoke-Checked (Find-HostTool 'ttt2-scan') @($xex, $ScanDir)
# XenonAnalyse's fixed patterns find no jump tables in TTT2 (its compiler schedules the
# dispatch instructions differently), so use our scheduling-tolerant detector instead.
Invoke-Checked python @('-I', (Join-Path $Root 'tools\find_switch_tables.py'), $ScanDir,
    (Join-Path $ConfigDir 'TTT2_switch_tables.toml'), (Join-Path $ConfigDir 'TTT2.toml'))

Write-Host "`nNext: copy the addresses from game\scan\recomp_addresses.toml into recomp\config\TTT2.toml," -ForegroundColor Green
Write-Host 'then run scripts\recompile.ps1' -ForegroundColor Green
