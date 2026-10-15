<#
.SYNOPSIS
  Installs the toolchain: CMake, Ninja, LLVM/Clang (>= 18), Emscripten (into .tools\emsdk).

.PARAMETER WithMsvc
  Also install Visual Studio 2022 Build Tools (C++ workload + Windows SDK). Clang on
  Windows needs the MSVC libraries and Windows SDK to link native executables
  (XenonRecomp, XenosRecomp, ttt2-scan). Skip it if you already have Visual Studio.
  Needs ~8 GB and an elevation (UAC) prompt.
#>
param([switch]$WithMsvc)
. "$PSScriptRoot\common.ps1"

$freeGb = [math]::Round((Get-PSDrive ($Root.Substring(0, 1))).Free / 1GB, 1)
Write-Host "Free space on drive: $freeGb GB"
if ($freeGb -lt 40) {
    Write-Warning ("The full pipeline needs roughly 40 GB: toolchains ~12 GB, recompiled C++ and " +
        "object files 10-20 GB, game dump ~8 GB.")
}

Require-Command winget 'Install "App Installer" from the Microsoft Store.'
$packages = @('Kitware.CMake', 'Ninja-build.Ninja', 'LLVM.LLVM')
if ($WithMsvc) { $packages += 'Microsoft.VisualStudio.2022.BuildTools' }

foreach ($id in $packages) {
    $wingetArgs = @('install', '--id', $id, '--exact', '--silent',
        '--accept-package-agreements', '--accept-source-agreements')
    if ($id -eq 'Microsoft.VisualStudio.2022.BuildTools') {
        $wingetArgs += @('--override', '--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended')
    }
    Write-Host "Installing $id ..." -ForegroundColor Cyan
    # winget returns non-zero when the package is already installed, so don't treat that as fatal.
    & winget @wingetArgs
}

# Emscripten lives inside the repo (.tools is gitignored) so it is pinned per checkout.
$emsdk = Join-Path $ToolsDir 'emsdk'
if (-not (Test-Path $emsdk)) {
    New-Item -ItemType Directory -Force $ToolsDir | Out-Null
    Invoke-Checked git @('clone', '--depth', '1', 'https://github.com/emscripten-core/emsdk.git', $emsdk)
}
Push-Location $emsdk
try {
    Invoke-Checked (Join-Path $emsdk 'emsdk.bat') @('install', 'latest')
    Invoke-Checked (Join-Path $emsdk 'emsdk.bat') @('activate', 'latest')
} finally { Pop-Location }

& "$PSScriptRoot\apply-patches.ps1"

Write-Host "`nDone. Open a NEW terminal so PATH changes apply, then run scripts\build-tools.ps1" -ForegroundColor Green
