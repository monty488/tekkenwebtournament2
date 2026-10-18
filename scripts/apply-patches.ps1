# Applies patches\<Submodule>-NNNN-*.patch to thirdparty\<Submodule>. Safe to re-run.
. "$PSScriptRoot\common.ps1"
Ensure-Submodules

Get-ChildItem (Join-Path $Root 'patches') -Filter '*.patch' | Sort-Object Name | ForEach-Object {
    $module = $_.Name.Split('-')[0]
    $dir = Join-Path $Third $module
    # Windows PowerShell turns native stderr into a terminating error under 'Stop';
    # a failing reverse-check is the expected "not applied yet" case.
    $ErrorActionPreference = 'Continue'
    & git -C $dir apply --reverse --check $_.FullName 2>$null | Out-Null
    $alreadyApplied = $LASTEXITCODE -eq 0
    $ErrorActionPreference = 'Stop'
    if ($alreadyApplied) {
        Write-Host "already applied: $($_.Name)"
    } else {
        Invoke-Checked git @('-C', $dir, 'apply', $_.FullName)
        Write-Host "applied: $($_.Name)" -ForegroundColor Green
    }
}
