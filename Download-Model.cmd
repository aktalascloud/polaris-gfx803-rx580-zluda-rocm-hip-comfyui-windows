@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\Download-Model.ps1" %*
if errorlevel 1 pause
