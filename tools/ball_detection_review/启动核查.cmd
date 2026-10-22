@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" review.py %*
if errorlevel 1 pause
exit /b
:missing
echo Run the environment installer in this folder first.
pause
exit /b 1
