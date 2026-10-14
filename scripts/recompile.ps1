# Runs XenonRecomp over default.xex, producing recomp\ppc\*.cpp, then regenerates
# the kernel import stubs for anything the runtime does not implement yet.
. "$PSScriptRoot\common.ps1"

$config = Join-Path $ConfigDir 'TTT2.toml'
if ((Get-Content $config -Raw) -match '(?m)^\w+_address\s*=\s*0x0+\s*$') {
    throw 'recomp\config\TTT2.toml still has 0x00000000 helper addresses. Run scripts\scan-xex.ps1 and copy them in.'
}

& "$PSScriptRoot\apply-patches.ps1"
New-Item -ItemType Directory -Force $PpcDir | Out-Null

$context = Join-Path $Third 'XenonRecomp\XenonUtils\ppc_context.h'
Invoke-Checked (Find-HostTool 'XenonRecomp') @($config, $context)

Invoke-Checked python @('-I', (Join-Path $PSScriptRoot 'gen_import_stubs.py'))

$count = (Get-ChildItem $PpcDir -Filter 'ppc_recomp.*.cpp').Count
Write-Host "`nGenerated $count translation units in recomp\ppc" -ForegroundColor Green
