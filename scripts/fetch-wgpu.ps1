<#
.SYNOPSIS
  Downloads the prebuilt wgpu-native library (WebGPU for the native build) into
  .tools\wgpu-native. The browser build uses Emscripten's emdawnwebgpu port instead;
  both implement the standard webgpu.h.
#>
param([string]$Version = 'v29.0.1.1')
. "$PSScriptRoot\common.ps1"

$dest = Join-Path $ToolsDir 'wgpu-native'
$stamp = Join-Path $dest 'VERSION'
if ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $Version)) {
    Write-Host "wgpu-native $Version already present in $dest"
    return
}

$asset = 'wgpu-windows-x86_64-msvc-release.zip'
$url = "https://github.com/gfx-rs/wgpu-native/releases/download/$Version/$asset"
$zip = Join-Path $env:TEMP "wgpu-native-$Version.zip"
Write-Host "Downloading $url ..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing

if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
New-Item -ItemType Directory -Force $dest | Out-Null
Expand-Archive -Path $zip -DestinationPath $dest
Remove-Item $zip
Set-Content -Path $stamp -Value $Version
Write-Host "wgpu-native $Version -> $dest" -ForegroundColor Green
