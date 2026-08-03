@echo off
REM ============================================
REM  Build script for Analyse Overselection .exe
REM ============================================
REM  Builds the application as a portable folder
REM  with interface.exe as the main entry point.
REM
REM  Prerequisites:
REM    pip install pyinstaller
REM
REM  Output: dist\AnalyseOverselection\interface.exe
REM ============================================

echo ============================================
echo  Building Analyse Overselection...
echo ============================================
echo.

REM Check PyInstaller is installed
where pyinstaller >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] PyInstaller not found. Installing...
    pip install pyinstaller
)

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

REM Build using spec file
echo.
echo Building executables...
pyinstaller build.spec --noconfirm

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the output above for details.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  BUILD SUCCESSFUL!
echo ============================================
echo.
echo  Output folder: dist\AnalyseOverselection\
echo  Main executable: dist\AnalyseOverselection\interface.exe
echo.
echo  To distribute:
echo    1. Copy the entire "dist\AnalyseOverselection" folder
echo    2. Users double-click "interface.exe" to launch
echo.
echo  Optional: create a shortcut to interface.exe
echo            and rename it "Analyse Overselection"
echo ============================================
pause
