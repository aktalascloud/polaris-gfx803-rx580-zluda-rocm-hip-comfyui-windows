# SPDX-License-Identifier: GPL-3.0-only
param(
    [ValidateSet('Comfy','PythonCuda')][string]$App='Comfy',
    [int]$Port=0,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ExtraArguments
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$profileFile=Join-Path $root 'profile.json'
if(!(Test-Path -LiteralPath $profileFile)){throw 'Run Install.cmd first. Launchers inside the installed runtime folder are the usable copies.'}
$backend=(Get-Content -LiteralPath $profileFile -Raw | ConvertFrom-Json).backend
if($backend -ne 'AMD'){throw 'This package targets AMD Polaris/gfx803 only'}
$python=Join-Path $root 'envs\cuda\python.exe'
$env:CUDA_MODULE_LOADING='LAZY'
$env:ZLUDA_WAVE64_SLOW_MODE='0'
$env:PYTHONUNBUFFERED='1'
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:DO_NOT_TRACK='1'
$env:TEMP=Join-Path $root 'temp'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
Remove-Item Env:HSA_OVERRIDE_GFX_VERSION -ErrorAction SilentlyContinue
Remove-Item Env:GPU_FORCE_BLIT_COPY_SIZE -ErrorAction SilentlyContinue
if($backend -eq 'AMD'){
    $env:HIP_PATH=Join-Path $root 'hip'
    $env:HIP_PLATFORM='amd';$env:HIP_COMPILER='clang';$env:HIP_RUNTIME='rocclr'
    $env:PATH=(Join-Path $root 'hip\bin')+';'+(Join-Path $root 'zluda')+';'+$env:PATH
}
if($App -eq 'Comfy'){
    if(!(Test-Path -LiteralPath (Join-Path $root 'apps\ComfyUI\main.py'))){throw 'ComfyUI was not selected during installation. Run Install.cmd -WithComfyUI with a fresh destination.'}
    if(!$Port){$Port=8188}
    $arguments=@('-I',(Join-Path $PSScriptRoot 'run_comfy.py'),'--listen','127.0.0.1','--port',"$Port",'--auto-launch','--force-fp32','--cpu-vae','--lowvram','--reserve-vram','0.9','--use-quad-cross-attention','--force-upcast-attention','--disable-cuda-malloc','--disable-xformers','--disable-api-nodes')
} else {
    $arguments=@('-I',(Join-Path $PSScriptRoot 'run_python.py'))
}
$arguments+=@($ExtraArguments)
& (Join-Path $root 'zluda\zluda.exe') -- $python @arguments
exit $LASTEXITCODE
