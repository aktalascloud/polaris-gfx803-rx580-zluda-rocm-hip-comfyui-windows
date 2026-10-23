@echo off
setlocal
if not exist "%~dp0profile.json" (
  echo Use the launcher inside the installed runtime folder. Run Install.cmd first.
  exit /b 1
)
set "TEMP=%~dp0temp"
set "TMP=%TEMP%"
set "CUDA_MODULE_LOADING=LAZY"
if not exist "%TEMP%" mkdir "%TEMP%"
findstr /c:"AMD" "%~dp0profile.json" >nul
if errorlevel 1 (
  "%~dp0envs\cuda\python.exe" -I "%~dp0tools\run_python.py" %*
) else (
  set "HIP_PATH=%~dp0hip"
  set "PATH=%~dp0hip\bin;%~dp0zluda;%PATH%"
  "%~dp0zluda\zluda.exe" -- "%~dp0envs\cuda\python.exe" -I "%~dp0tools\run_python.py" %*
)
exit /b %ERRORLEVEL%
