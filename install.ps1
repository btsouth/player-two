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
        # In its own PowerShell, so a failure inside uv's installer can't close this window.
        & powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" *> $null
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

    # Step 3: what needs administrator rights, behind a single Windows permission prompt.
    # The Visual C++ runtime, which Player Two's voice needs and clean PCs often lack, and
    # the ViGEmBus driver for the virtual controller (not on ARM, which it doesn't support).
    $needVc = -not (Test-Path "$env:WINDIR\System32\msvcp140.dll") -or
              -not (Test-Path "$env:WINDIR\System32\msvcp140_1.dll") -or
              -not (Test-Path "$env:WINDIR\System32\vcruntime140_1.dll")
    $needPad = -not $arm -and -not (Get-Service ViGEmBus -ErrorAction SilentlyContinue)
    $msi = $null
    if ($needPad) {
        $tools = (& uv tool dir).Trim()
        $msi = Get-ChildItem "$tools\player-two" -Recurse -Filter "ViGEmBusSetup_x64.msi" -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $msi) { $needPad = $false; Say "I couldn't find the controller driver installer. I'll press keyboard keys instead." }
    }
    $vcExe = Join-Path $env:TEMP "vc_redist.exe"
    if ($needVc) {
        $vcArch = if ($arm) { "arm64" } else { "x64" }
        try {
            Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.$vcArch.exe" -OutFile $vcExe -UseBasicParsing
        } catch {
            $needVc = $false
            Say "I couldn't download the Visual C++ runtime. If my voice doesn't work, install it from aka.ms/vs/17/release/vc_redist.$vcArch.exe."
        }
    }
    if (-not $needVc -and -not $needPad) {
        Say "Step 3 of 3: nothing else to install."
    } else {
        $what = @()
        if ($needVc) { $what += "the Visual C++ runtime my voice needs" }
        if ($needPad) { $what += "the ViGEmBus driver, so I can press buttons as a second controller" }
        Say "Step 3 of 3: installing $($what -join ', and '). Windows will ask for permission once; choose Yes."
        $codes = Join-Path $env:TEMP "player-two-codes.txt"
        Remove-Item $codes -ErrorAction SilentlyContinue
        $lines = @('$c = @()')
        if ($needVc) { $lines += "`$c += 'vc ' + (Start-Process '$vcExe' -ArgumentList '/install','/quiet','/norestart' -Wait -PassThru).ExitCode" }
        if ($needPad) { $lines += "`$c += 'pad ' + (Start-Process msiexec.exe -ArgumentList '/i','`"$($msi.FullName)`"','/passive','/norestart' -Wait -PassThru).ExitCode" }
        $lines += "Set-Content -Path '$codes' -Value `$c"
        $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($lines -join "`n"))
        try {
            Start-Process powershell.exe -Verb RunAs -Wait -WindowStyle Hidden `
                -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-EncodedCommand",$encoded
            $results = @{}
            Get-Content $codes -ErrorAction Stop | ForEach-Object { $k, $v = $_ -split " "; $results[$k] = [int]$v }
            if ($needVc) {
                if ($results["vc"] -in 0, 1638, 3010) { Say "The Visual C++ runtime is installed." }
                else { Say "The Visual C++ runtime didn't install (code $($results["vc"])). If my voice doesn't work, run this installer again." }
            }
            if ($needPad) {
                if ($results["pad"] -in 0, 3010) { Say "The controller driver is installed." }
                else { Say "The controller driver didn't install (code $($results["pad"])). I'll press keyboard keys instead." }
            }
        } catch {
            Say "Nothing was installed, because permission wasn't given. Run this installer again to add it later. Until then, I'll press keyboard keys instead of a controller, and my voice may not work."
        }
    }
    if ($arm) { Say "The controller driver doesn't support ARM computers, so I'll press keyboard keys instead." }

    Say "Done. Now a few questions to set things up."
    & player-two setup
}

Install-PlayerTwo
