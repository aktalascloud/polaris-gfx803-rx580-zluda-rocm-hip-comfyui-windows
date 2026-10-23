@echo off
setlocal
if not exist "%~dp0profile.json" exit /b 1
if "%~1"=="" (
  echo Usage: Run-ZLUDA.cmd program.exe [arguments]
  exit /b 1
)
set "HIP_PATH=%~dp0hip"
set "ROCBLAS_TENSILE_LIBPATH=%~dp0hip\bin\rocblas\library"
set "PATH=%~dp0hip\bin;%~dp0zluda;%PATH%"
set "DISABLE_ADDMM_CUDA_LT=1"
set "HSA_OVERRIDE_GFX_VERSION="
set "GPU_FORCE_BLIT_COPY_SIZE="
"%~dp0zluda\zluda.exe" -- %*
exit /b %ERRORLEVEL%
