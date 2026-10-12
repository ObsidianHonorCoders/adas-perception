@echo off
setlocal
cd /d "%~dp0"
set "RESULT=1"

where py >nul 2>&1
if errorlevel 1 (
    echo Python Launcher for Windows ^(py^) was not found.
    echo Install Python 3.10 and the Python Launcher, then run this script again.
    goto :finish
)

py -3.10 -m venv .venv
if errorlevel 1 (
    echo Could not create .venv with Python 3.10.
    echo Confirm Python 3.10 is installed by running: py -0p
    goto :finish
)

call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo Could not activate .venv.
    goto :finish
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo Failed to upgrade pip.
    goto :finish
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    goto :finish
)

python ..\scripts\download_weights.py
if errorlevel 1 (
    echo Dependencies were installed, but downloading YOLO11n weights failed.
    goto :finish
)

set "RESULT=0"

:finish
if "%RESULT%"=="0" echo Environment is ready. Activate it with: .venv\Scripts\activate
pause
exit /b %RESULT%
