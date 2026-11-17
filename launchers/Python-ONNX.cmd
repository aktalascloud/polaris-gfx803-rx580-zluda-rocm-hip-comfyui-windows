@echo off
"%~dp0envs\onnx\python.exe" %*
if errorlevel 1 pause
