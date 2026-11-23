@echo off
setlocal DisableDelayedExpansion
if not exist "%~dp0profile.json" exit /b 1
if /i "%~1"=="--torch" (
    call "%~dp0Python-CUDA.cmd" "%~dp0tools\check_torch.py"
    exit /b
)
set "HIP_PATH=%~dp0hip"
set "ROCBLAS_TENSILE_LIBPATH=%~dp0hip\bin\rocblas\library"
set "PATH=%~dp0hip\bin;%~dp0zluda;%PATH%"
"%~dp0envs\diagnostics\python.exe" -I "%~dp0tools\check_cuda.py" --zluda-bin "%~dp0zluda" --root "%~dp0." --report "%~dp0temp\cuda-driver-api.json" %*
exit /b %ERRORLEVEL%
