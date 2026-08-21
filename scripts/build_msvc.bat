@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo Brass MSVC Toolchain Build ^& Test
echo ======================================================================

set "VCVARS="
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else (
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "!VSWHERE!" (
        for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
            )
        )
    )
)

if "%VCVARS%"=="" (
    echo [ERROR] Could not locate MSVC vcvars64.bat. Please ensure Visual Studio 2022 is installed.
    exit /b 1
)

echo [INFO] Using MSVC environment: %VCVARS%
call "%VCVARS%"
if errorlevel 1 (
    echo [ERROR] Failed to initialize MSVC x64 environment.
    exit /b 1
)

cd /d "%~dp0\.."

set "BUILD_DIR=build_msvc"

echo [INFO] Configuring CMake with MSVC (/W4 /WX)...
where ninja >nul 2>nul
if %errorlevel% equ 0 (
    cmake -B "%BUILD_DIR%" -G "Ninja" -DCMAKE_BUILD_TYPE=Release
) else (
    cmake -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64
)

if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)

echo [INFO] Building Brass (/W4 /WX clean)...
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo [INFO] Running CTest suite...
ctest --test-dir "%BUILD_DIR%" --output-on-failure -C Release -R unit_tests
if errorlevel 1 (
    echo [ERROR] Tests failed.
    exit /b 1
)

echo ======================================================================
echo [SUCCESS] Brass built and verified cleanly under MSVC (/W4 /WX)!
echo ======================================================================
exit /b 0
