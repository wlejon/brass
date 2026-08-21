<#
.SYNOPSIS
    Build and test Brass using the MSVC toolchain (/W4 /WX clean).
#>

[CmdletBinding()]
param (
    [string]$BuildType = "Release",
    [string]$BuildDir = "build_msvc",
    [switch]$RunTests = $true
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Brass MSVC Toolchain Build & Test (PowerShell)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$candidates = @(
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "$env:ProgramFiles\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
)

$vcvars = $null
foreach ($c in $candidates) {
    if (Test-Path $c) {
        $vcvars = $c
        break
    }
}

if (-not $vcvars) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath -and (Test-Path "$installPath\VC\Auxiliary\Build\vcvars64.bat")) {
            $vcvars = "$installPath\VC\Auxiliary\Build\vcvars64.bat"
        }
    }
}

if (-not $vcvars) {
    throw "Could not locate MSVC vcvars64.bat. Please ensure Visual Studio 2022 is installed."
}

Write-Host "[INFO] Using MSVC environment: $vcvars" -ForegroundColor Green

# Function to invoke a batch command inside the vcvars64 environment
function Invoke-MsvcCommand {
    param([string]$Cmd)
    $tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
    try {
        $content = "@call `"$vcvars`" >nul 2>nul`r`n$Cmd"
        Set-Content -Path $tempBat -Value $content -Encoding ASCII
        cmd.exe /c $tempBat
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code $($LASTEXITCODE) - $Cmd"
        }
    } finally {
        if (Test-Path $tempBat) { Remove-Item $tempBat -Force -ErrorAction SilentlyContinue }
    }
}

$repoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $repoRoot

$hasNinja = (Get-Command ninja -ErrorAction SilentlyContinue) -ne $null
$generator = if ($hasNinja) { "-G Ninja" } else { "-G `"Visual Studio 17 2022`" -A x64" }

Write-Host "[INFO] Configuring CMake with MSVC (/W4 /WX)..." -ForegroundColor Green
Invoke-MsvcCommand "cmake -B `"$BuildDir`" $generator -DCMAKE_BUILD_TYPE=$BuildType"

Write-Host "[INFO] Building Brass (/W4 /WX clean)..." -ForegroundColor Green
Invoke-MsvcCommand "cmake --build `"$BuildDir`" --config $BuildType"

if ($RunTests) {
    Write-Host "[INFO] Running CTest suite..." -ForegroundColor Green
    Invoke-MsvcCommand "ctest --test-dir `"$BuildDir`" --output-on-failure -C $BuildType"
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "[SUCCESS] Brass built and verified cleanly under MSVC (/W4 /WX)!" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
