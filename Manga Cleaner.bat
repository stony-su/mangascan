@echo off
rem Double-click to open the Manga Cleaner review app in your browser.
setlocal
cd /d "%~dp0"
title Manga Cleaner

if not exist ".venv\Scripts\python.exe" call :setup
if not exist ".venv\Scripts\python.exe" goto :failed

".venv\Scripts\python.exe" -m mangaclean --ui %*
if errorlevel 1 pause
exit /b

:setup
echo First run: setting up. This needs an internet connection and takes a minute or two...
where py >nul 2>nul
if not errorlevel 1 (py -3 -m venv .venv) else (python -m venv .venv)
if not exist ".venv\Scripts\python.exe" exit /b 1
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
  echo Installing the required packages failed.
  rmdir /s /q .venv
)
exit /b

:failed
echo.
echo Could not set up Python. Install Python 3.10 or newer from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" in the installer) and run this file again.
pause
exit /b 1
