# Configures and builds the WebAssembly target into build\web-<config>.
param([ValidateSet('Debug', 'Release', 'RelWithDebInfo')][string]$Config = 'Release')
. "$PSScriptRoot\common.ps1"

if (-not (Get-Command emcmake -ErrorAction SilentlyContinue)) {
    $emsdkEnv = Join-Path $ToolsDir 'emsdk\emsdk_env.ps1'
    if (-not (Test-Path $emsdkEnv)) { throw 'Emscripten not installed. Run scripts\bootstrap.ps1' }
    . $emsdkEnv
}

$out = Join-Path $BuildDir "web-$($Config.ToLower())"
Invoke-Checked emcmake @('cmake', '-S', $Root, '-B', $out, '-G', 'Ninja', "-DCMAKE_BUILD_TYPE=$Config")
Invoke-Checked cmake @('--build', $out, '--parallel')
Write-Host "`nBuilt. Serve with: node web\serve.mjs $out" -ForegroundColor Green
