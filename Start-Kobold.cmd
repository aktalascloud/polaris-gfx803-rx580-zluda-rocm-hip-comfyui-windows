@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\Launch.ps1" -App Kobold %*
if errorlevel 1 pause
