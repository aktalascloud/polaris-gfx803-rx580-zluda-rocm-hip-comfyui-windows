@echo off
setlocal
rem Windows PowerShell must resolve its own built-in modules when called from pwsh.
set "PSModulePath="
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\Install.ps1" %*
set "result=%ERRORLEVEL%"
if not "%result%"=="0" echo Installation failed. Review the error above.
if "%~1"=="" pause
exit /b %result%
