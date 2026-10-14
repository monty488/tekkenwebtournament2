# Builds the host-side tools natively with clang-cl + Ninja:
#   XenonAnalyse, XenonRecomp, ttt2-scan, XenosRecomp, extract-xiso
. "$PSScriptRoot\common.ps1"
Require-Command cmake 'Run scripts\bootstrap.ps1'
Require-Command ninja 'Run scripts\bootstrap.ps1'
Require-Command clang-cl 'Run scripts\bootstrap.ps1 (LLVM)'

& "$PSScriptRoot\apply-patches.ps1"

function Build-Project([string]$Source, [string]$Name) {
    $out = Join-Path $BuildDir "tools\$Name"
    Invoke-Checked cmake @('-S', $Source, '-B', $out, '-G', 'Ninja',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_C_COMPILER=clang-cl', '-DCMAKE_CXX_COMPILER=clang-cl')
    Invoke-Checked cmake @('--build', $out, '--parallel')
}

Build-Project (Join-Path $Root 'tools') 'xenon'
Build-Project (Join-Path $Third 'XenosRecomp') 'xenos'
Build-Project (Join-Path $Third 'extract-xiso') 'extract-xiso'

Write-Host "`nTools built under build\tools" -ForegroundColor Green
