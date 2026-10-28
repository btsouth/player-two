# Player Two installer for Windows. Run in PowerShell:
#   irm https://raw.githubusercontent.com/btsouth/player-two/main/install.ps1 | iex
# It installs uv (if needed), Player Two, and the ViGEmBus controller driver, then starts
# setup. Every step prints one plain line, so it reads well in a screen reader, and a
# failure never closes the window.

# Native programs (uv) write progress to stderr; in Windows PowerShell 5.1 "Stop" would turn
# that into a fatal error. Check exit codes instead.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"

function Say($text) { Write-Host $text }

function Install-PlayerTwo {
    $source = if ($env:PLAYER_TWO_SOURCE) { $env:PLAYER_TWO_SOURCE } else {
        "https://github.com/btsouth/player-two/archive/refs/heads/main.zip" }
    $arm = $env:PROCESSOR_ARCHITECTURE -eq "ARM64"

    Say "Installing Player Two. This takes a few minutes."

    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Say "Step 1 of 3: installing uv, the Python tool installer."
        try {
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression *> $null
        } catch {
            Say "Installing uv failed: $($_.Exception.Message). Check the internet connection and run this again."
            return
        }
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Say "Installing uv failed. Check the internet connection and run this again."
            return
        }
    } else {
        Say "Step 1 of 3: uv is already installed."
    }

    Say "Step 2 of 3: installing Player Two."
    # vgamepad would otherwise open the driver installer's full wizard, hidden behind this
    # window, in the middle of the install; step 3 installs the driver quietly instead.
    $env:VGAMEPAD_SKIP_VIGEMBUS_INSTALL = "true"
    $log = Join-Path $env:TEMP "player-two-install.log"
    & uv tool install --force --python 3.12 "player-two[windows] @ $source" *> $log
    if ($LASTEXITCODE -ne 0) {
        Say "Installing Player Two failed. The details are in $log. Run this again, or report it at github.com/btsouth/player-two/issues."
        return
    }
    & uv tool update-shell *> $null
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

    if ($arm) {
        Say "Step 3 of 3: skipped. The controller driver doesn't support ARM computers, so I'll press keyboard keys instead."
    } elseif (Get-Service ViGEmBus -ErrorAction SilentlyContinue) {
        Say "Step 3 of 3: the controller driver is already installed."
    } else {
        Say "Step 3 of 3: installing the ViGEmBus controller driver, so I can press buttons as a second controller. Windows will ask for permission; choose Yes."
        $tools = (& uv tool dir).Trim()
        $msi = Get-ChildItem "$tools\player-two" -Recurse -Filter "ViGEmBusSetup_x64.msi" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $msi) {
            Say "I couldn't find the controller driver installer. I'll press keyboard keys instead."
        } else {
            try {
                $p = Start-Process msiexec.exe -Verb RunAs -Wait -PassThru `
                    -ArgumentList "/i `"$($msi.FullName)`" /passive /norestart"
                if ($p.ExitCode -eq 0 -or $p.ExitCode -eq 3010) {
                    Say "The controller driver is installed."
                } else {
                    Say "The controller driver didn't install (code $($p.ExitCode)). I'll press keyboard keys instead."
                }
            } catch {
                Say "The controller driver wasn't installed, because permission wasn't given. I'll press keyboard keys instead. Run this installer again to add it later."
            }
        }
    }

    Say "Done. Now a few questions to set things up."
    & player-two setup
}

Install-PlayerTwo
