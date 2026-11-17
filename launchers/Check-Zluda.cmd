@echo off
setlocal
if not exist "%~dp0profile.json" exit /b 1
set "HIP_PATH=%~dp0hip"
set "ROCBLAS_TENSILE_LIBPATH=%~dp0hip\bin\rocblas\library"
set "PATH=%~dp0hip\bin;%~dp0zluda;%PATH%"
"%~dp0envs\diagnostics\python.exe" -I "%~dp0tools\check_cuda.py" --zluda-bin "%~dp0zluda" --root "%~dp0." --report "%~dp0temp\cuda-driver-api.json" %*
exit /b %ERRORLEVEL%
