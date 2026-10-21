@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (
 echo Installation failed. Check your conda environment.
 pause
 exit /b 1
)
echo All dependencies installed.
pause
