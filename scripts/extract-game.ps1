<#
.SYNOPSIS
  Extracts your Tekken Tag Tournament 2 disc image into game\files.
.EXAMPLE
  scripts\extract-game.ps1 -Iso "D:\Dumps\TTT2.iso"
#>
param([Parameter(Mandatory)][string]$Iso)
. "$PSScriptRoot\common.ps1"

if (-not (Test-Path $Iso)) { throw "ISO not found: $Iso" }
$xiso = Find-HostTool 'extract-xiso'
$files = Join-Path $GameDir 'files'
New-Item -ItemType Directory -Force $files | Out-Null

Invoke-Checked $xiso @('-x', (Resolve-Path $Iso).Path, '-d', $files)

if (-not (Test-Path (Join-Path $files 'default.xex'))) {
    throw ("default.xex not found after extraction. If your dump is a Games-on-Demand (GOD) " +
        "container rather than an ISO, convert it to ISO first.")
}
Write-Host "Extracted to $files" -ForegroundColor Green
