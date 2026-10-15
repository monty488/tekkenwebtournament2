# Runs XenonRecomp over default.xex, producing recomp\ppc\*.cpp, then regenerates
# the kernel import stubs for anything the runtime does not implement yet.
. "$PSScriptRoot\common.ps1"

$config = Join-Path $ConfigDir 'TTT2.toml'
if ((Get-Content $config -Raw) -match '(?m)^\w+_address\s*=\s*0x0+\s*$') {
    throw 'recomp\config\TTT2.toml still has 0x00000000 helper addresses. Run scripts\scan-xex.ps1 and copy them in.'
}

& "$PSScriptRoot\apply-patches.ps1"

# Generate into an empty directory: when a run produces fewer translation units, the old
# extra ppc_recomp.N.cpp files would otherwise linger and get compiled (duplicate symbols).
# Files whose content is unchanged get their previous timestamp back so Ninja skips them.
$prevDir = "$PpcDir.prev"
if (Test-Path $prevDir) { Remove-Item -Recurse -Force $prevDir }
if (Test-Path $PpcDir) { Rename-Item $PpcDir $prevDir }
New-Item -ItemType Directory -Force $PpcDir | Out-Null

$context = Join-Path $Third 'XenonRecomp\XenonUtils\ppc_context.h'
Invoke-Checked (Find-HostTool 'XenonRecomp') @($config, $context)

if (Test-Path $prevDir) {
    foreach ($file in Get-ChildItem $PpcDir -File) {
        $old = Join-Path $prevDir $file.Name
        if ((Test-Path $old) -and ((Get-FileHash $old).Hash -eq (Get-FileHash $file.FullName).Hash)) {
            $file.LastWriteTime = (Get-Item $old).LastWriteTime
        }
    }
    Remove-Item -Recurse -Force $prevDir
}

Invoke-Checked python @('-I', (Join-Path $PSScriptRoot 'gen_import_stubs.py'))

$count = (Get-ChildItem $PpcDir -Filter 'ppc_recomp.*.cpp').Count
Write-Host "`nGenerated $count translation units in recomp\ppc" -ForegroundColor Green
