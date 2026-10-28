# Player Two installer for Windows. Run in PowerShell:
#   irm https://raw.githubusercontent.com/btsouth/player-two/main/install.ps1 | iex
# It installs uv (if needed), Player Two, and the ViGEmBus controller driver, then starts
# setup. Every step prints one plain line, so it reads well in a screen reader.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$source = if ($env:PLAYER_TWO_SOURCE) { $env:PLAYER_TWO_SOURCE } else {
    "https://github.com/btsouth/player-two/archive/refs/heads/main.zip" }

function Say($text) { Write-Host $text }

Say "Installing Player Two. This takes a few minutes."

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Say "Step 1 of 3: installing uv, the Python tool installer."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression | Out-Null
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
} else {
    Say "Step 1 of 3: uv is already installed."
}

Say "Step 2 of 3: installing Player Two."
uv tool install --force --python 3.12 "player-two[windows] @ $source" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Say "Installing Player Two failed. Run this again, or report it at github.com/btsouth/player-two/issues."
    exit 1
}
uv tool update-shell 2>&1 | Out-Null
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

if (Get-Service ViGEmBus -ErrorAction SilentlyContinue) {
    Say "Step 3 of 3: the controller driver is already installed."
} else {
    Say "Step 3 of 3: installing the ViGEmBus controller driver, so I can press buttons as a second controller. Windows will ask for permission; choose Yes."
    $tools = (uv tool dir).Trim()
    $msi = Get-ChildItem "$tools\player-two" -Recurse -Filter "ViGEmBusSetup_x64.msi" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($msi) {
        $p = Start-Process msiexec.exe -Verb RunAs -Wait -PassThru `
            -ArgumentList "/i `"$($msi.FullName)`" /passive /norestart"
        if ($p.ExitCode -eq 0 -or $p.ExitCode -eq 3010) {
            Say "The controller driver is installed."
        } else {
            Say "The controller driver didn't install. I'll press keyboard keys instead."
        }
    } else {
        Say "I couldn't find the controller driver installer. I'll press keyboard keys instead."
    }
}

Say "Done. Now a few questions to set things up."
player-two setup
