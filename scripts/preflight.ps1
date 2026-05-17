# scripts/preflight.ps1 -- Windows wrapper for preflight_validate.py.
#
# Sets the UTF-8 environment FIRST so the validator (and any modal CLI
# subprocesses it spawns) don't crash on the charmap codec, then shells
# out to scripts/preflight_validate.py. Returns the validator's exit
# code unchanged: 0 = proceed, 1 = infra fail, 2 = cost ceiling breach.
#
# Usage:
#   .\scripts\preflight.ps1 -Config configs\all_models.yaml
#   .\scripts\preflight.ps1 -Config configs\cost_controlled_modal.yaml -AutoCreateVolume
#   .\scripts\preflight.ps1 -Config configs\cost_controlled_modal.yaml -RemoteSmoke
#   .\scripts\preflight.ps1 -Config configs\cost_controlled_modal.yaml -Json
#   .\scripts\preflight.ps1 -Config <path> -Skip COST-1,CODE-3

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Config,
    [string]$Volume = "modallabs-runs",
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$AutoCreateVolume,
    [switch]$AllowLongTimeout,
    [switch]$RemoteSmoke,
    [switch]$Json,
    [string[]]$Skip = @()
)

# UTF-8 stdio: ALWAYS set before any modal command. PowerShell's default
# codec on Windows is cp1252; modal CLI emits UTF-8 checkmarks ("✓") and
# crashes mid-upload without these env vars in place.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# PATH: prepend the user-site Python Scripts dir so `modal.exe` resolves
# even if it was installed via `pip install --user modal`. The validator
# falls back to a hard-coded probe too, so this is belt-and-braces.
$pyVer = & python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
$scriptsDir = Join-Path $env:APPDATA "Python\Python$pyVer\Scripts"
if (Test-Path $scriptsDir) {
    if ($env:PATH -notlike "*$scriptsDir*") {
        $env:PATH = "$scriptsDir;$env:PATH"
    }
}

# Translate switches to validator flags.
$pyArgs = @(
    "$PSScriptRoot\preflight_validate.py",
    "--config", $Config,
    "--repo-root", $RepoRoot,
    "--volume", $Volume
)
if ($AutoCreateVolume) { $pyArgs += "--auto-create-volume" }
if ($AllowLongTimeout) { $pyArgs += "--allow-long-timeout" }
if ($RemoteSmoke)      { $pyArgs += "--remote-smoke" }
if ($Json)             { $pyArgs += "--json" }
if ($Skip -and $Skip.Count -gt 0) {
    $pyArgs += "--skip"
    $pyArgs += $Skip
}

& python @pyArgs
exit $LASTEXITCODE
