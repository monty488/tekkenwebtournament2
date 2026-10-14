# Runs ttt2-scan (image dump, imports, helper addresses) and XenonAnalyse (jump tables).
. "$PSScriptRoot\common.ps1"

$xex = Join-Path $GameDir 'files\default.xex'
$patched = Join-Path $GameDir 'update\default_patched.xex'
if (Test-Path $patched) { $xex = $patched; Write-Host 'Using title-updated XEX' }
if (-not (Test-Path $xex)) { throw "Missing $xex. Run scripts\extract-game.ps1 first." }

Invoke-Checked (Find-HostTool 'ttt2-scan') @($xex, $ScanDir)
Invoke-Checked (Find-HostTool 'XenonAnalyse') @($xex, (Join-Path $ConfigDir 'TTT2_switch_tables.toml'))

Write-Host "`nNext: copy the addresses from game\scan\recomp_addresses.toml into recomp\config\TTT2.toml," -ForegroundColor Green
Write-Host 'then run scripts\recompile.ps1' -ForegroundColor Green
