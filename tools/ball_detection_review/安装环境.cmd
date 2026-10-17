@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Environment ready.
pause
exit /b 0
:failed
echo Installation failed. Check Python 3.12 and the error above.
pause
exit /b 1
