<#
.SYNOPSIS
  Downloads FFmpeg's source into .tools\ffmpeg and prepares a minimal build of its XMA
  decoders (libavcodec + libavutil, nothing else) for the native and/or web build.

.DESCRIPTION
  FFmpeg's own configure script (run under Git Bash's sh) writes config.h for each target
  with every component disabled except the xma1/xma2 decoders. tools\ffmpeg_sources.py then
  reads FFmpeg's Makefiles for that configuration and writes the list of C files, which
  runtime\CMakeLists.txt compiles with the project's own toolchain (no make needed).

  Targets: .tools\ffmpeg\native (clang-cl / MSVC) and .tools\ffmpeg\web (emscripten).
  FFmpeg is LGPL; it is fetched, never committed.
#>
param(
    [string]$Version = 'n7.1.1',
    [ValidateSet('native', 'web', 'all')][string]$Target = 'all'
)
. "$PSScriptRoot\common.ps1"

$base = Join-Path $ToolsDir 'ffmpeg'
$src = Join-Path $base 'src'
$stamp = Join-Path $src 'VERSION_FETCHED'

if (-not ((Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $Version))) {
    $url = "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/$Version.zip"
    $zip = Join-Path $env:TEMP "ffmpeg-$Version.zip"
    Write-Host "Downloading $url ..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    if (Test-Path $src) { Remove-Item -Recurse -Force $src }
    New-Item -ItemType Directory -Force $base | Out-Null
    $unpack = Join-Path $base 'unpack'
    if (Test-Path $unpack) { Remove-Item -Recurse -Force $unpack }
    Expand-Archive -Path $zip -DestinationPath $unpack
    Move-Item (Get-ChildItem $unpack | Select-Object -First 1).FullName $src
    Remove-Item -Recurse -Force $unpack
    Remove-Item $zip
    Set-Content -Path $stamp -Value $Version
    Write-Host "FFmpeg $Version -> $src" -ForegroundColor Green
} else {
    Write-Host "FFmpeg $Version already present in $src"
}

# configure runs under Git Bash's sh; only its generated headers and config.mak are used.
$bash = @("$env:ProgramFiles\Git\bin\bash.exe", "$env:ProgramFiles\Git\usr\bin\bash.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $bash) { throw 'Git Bash not found (FFmpeg''s configure needs a POSIX shell).' }
function To-Posix([string]$p) { '/' + $p.Substring(0, 1).ToLower() + ($p.Substring(2) -replace '\\', '/') }

$common = '--disable-all --disable-autodetect --enable-avcodec --enable-avutil --enable-decoder=xma1,xma2 ' +
          '--disable-asm --disable-pthreads --disable-w32threads --disable-debug --disable-doc --disable-programs'

function Configure-Target([string]$Name, [string]$Setup, [string]$Options) {
    $out = Join-Path $base $Name
    New-Item -ItemType Directory -Force $out | Out-Null
    Write-Host "Configuring FFmpeg ($Name) ..." -ForegroundColor Cyan
    $script = "$Setup cd '$(To-Posix $out)' && '$(To-Posix $src)/configure' $common $Options > configure.log 2>&1"
    & $bash -lc $script
    if ($LASTEXITCODE -ne 0) {
        # configure.log only says that a test failed; config.log says why.
        $details = Join-Path $out 'ffbuild\config.log'
        if (Test-Path $details) {
            Write-Host "--- last lines of $details ---" -ForegroundColor Yellow
            Get-Content $details -Tail 30 | ForEach-Object { Write-Host $_ }
        }
        throw "FFmpeg configure ($Name) failed: see $out\configure.log and $details"
    }
    Invoke-Checked python @('-I', (Join-Path $PSScriptRoot 'ffmpeg_sources.py'), $src, $out)
}

if ($Target -in 'native', 'all') {
    Enter-MsvcEnv
    Configure-Target 'native' '' '--toolchain=msvc --cc=clang-cl --cxx=clang-cl'
}
if ($Target -in 'web', 'all') {
    $emsdk = Join-Path $ToolsDir 'emsdk'
    if (-not (Test-Path (Join-Path $emsdk 'emsdk_env.sh'))) { throw 'emsdk not found in .tools\emsdk. Run scripts\bootstrap.ps1' }
    # wasm64 (memory64), as the runtime: pointer sizes are baked into config.h.
    Configure-Target 'web' "source '$(To-Posix $emsdk)/emsdk_env.sh' > /dev/null 2>&1 && " `
        ('--enable-cross-compile --target-os=none --arch=wasm --cc=emcc --cxx=em++ --host-cc=emcc --ar=emar --ranlib=emranlib --nm=emnm ' +
         '--extra-cflags=-sMEMORY64=1 --extra-ldflags=-sMEMORY64=1')
}
Write-Host 'FFmpeg ready.' -ForegroundColor Green
