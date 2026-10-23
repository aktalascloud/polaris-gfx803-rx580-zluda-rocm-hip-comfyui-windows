@echo off
setlocal
if not exist "%~dp0envs\torch-directml\python.exe" (
  echo Install with -WithTorchDirectML first, then use the launcher inside runtime.
  exit /b 1
)
"%~dp0envs\torch-directml\python.exe" -I "%~dp0tools\run_directml.py" %*
exit /b %ERRORLEVEL%
