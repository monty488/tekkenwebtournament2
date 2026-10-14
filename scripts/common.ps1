# Shared paths/helpers for the pipeline scripts. Dot-source it: . "$PSScriptRoot\common.ps1"
$ErrorActionPreference = 'Stop'

$Root       = Split-Path -Parent $PSScriptRoot
$Third      = Join-Path $Root 'thirdparty'
$ToolsDir   = Join-Path $Root '.tools'
$BuildDir   = Join-Path $Root 'build'
$GameDir    = Join-Path $Root 'game'
$ScanDir    = Join-Path $GameDir 'scan'
$ConfigDir  = Join-Path $Root 'recomp\config'
$PpcDir     = Join-Path $Root 'recomp\ppc'

# Make tools installed by bootstrap.ps1 visible even if this shell predates them.
foreach ($p in @("$env:ProgramFiles\CMake\bin", "$env:ProgramFiles\LLVM\bin")) {
    if ((Test-Path $p) -and -not (($env:Path -split ';') -contains $p)) { $env:Path = "$p;$env:Path" }
}

function Require-Command([string]$Name, [string]$Hint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "'$Name' not found. $Hint"
    }
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited with code $LASTEXITCODE" }
}

function Find-HostTool([string]$Name) {
    $hit = Get-ChildItem -Path (Join-Path $BuildDir 'tools') -Recurse -Filter "$Name.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $hit) { throw "$Name.exe not built yet. Run scripts\build-tools.ps1 first." }
    return $hit.FullName
}
