@echo off
rem Drag a folder, some images, or a .zip/.cbz onto this file to clean the pages,
rem translate them to English and letter them, without opening the app.
rem Results go into a "translated" folder next to the input, with a .txt script
rem of every page (the Japanese that was read and its translation).
rem Translation uses Claude through your Claude Code login when it is installed
rem (each page is sent to Anthropic), otherwise a rough offline draft.
rem A notes.txt in the folder tells the translator about the series.
setlocal
cd /d "%~dp0"
title Manga Cleaner - translate

if "%~1"=="" (
  echo Drag a folder of raw pages, image files, or a .zip / .cbz onto this file.
  echo The English pages are written to a "translated" folder next to them.
  echo.
  pause
  exit /b
)

if not exist ".venv\Scripts\python.exe" call :setup
if not exist ".venv\Scripts\python.exe" goto :failed

".venv\Scripts\python.exe" -m mangaclean --translate --script %*
echo.
pause
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
