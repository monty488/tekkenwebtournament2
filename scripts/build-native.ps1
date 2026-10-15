# Native Windows build of the same runtime. Bring the game up here first:
# debuggers, sanitizers and crash dumps are far better than in a browser.
param([ValidateSet('Debug', 'Release', 'RelWithDebInfo')][string]$Config = 'Debug')
. "$PSScriptRoot\common.ps1"
Require-Command clang-cl 'Run scripts\bootstrap.ps1'
Enter-MsvcEnv

$out = Join-Path $BuildDir "native-$($Config.ToLower())"
Invoke-Checked cmake @('-S', $Root, '-B', $out, '-G', 'Ninja', "-DCMAKE_BUILD_TYPE=$Config",
    '-DCMAKE_C_COMPILER=clang-cl', '-DCMAKE_CXX_COMPILER=clang-cl')
Invoke-Checked cmake @('--build', $out, '--parallel')
