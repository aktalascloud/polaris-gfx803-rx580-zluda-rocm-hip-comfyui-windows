# SPDX-License-Identifier: GPL-3.0-only
param([string]$ModelDirectory='')
$ErrorActionPreference='Stop'
$parent=Split-Path $PSScriptRoot -Parent
if(!$ModelDirectory){
    if(Test-Path -LiteralPath (Join-Path $parent 'profile.json')){$ModelDirectory=Join-Path $parent 'models'}
    else{$ModelDirectory=Join-Path $parent 'runtime\models'}
}
$name='Qwen3-4B-Instruct-2507-heretic-uncens.i1-IQ4_XS.gguf'
$url='https://huggingface.co/mradermacher/Qwen3-4B-Instruct-2507-heretic-uncens-i1-GGUF/resolve/8892eb7be59811232116c04f191503d7ec823c27/'+$name
$expected='78614724ef8ccb9d50831fb24b3be3b6a90a4b7c5d73230531fdb80e4386a98d'
Write-Host 'Optional 2.27 GB model from Hugging Face, not part of this release.'
Write-Host 'Read the upstream model card and Apache-2.0 terms before downloading:'
Write-Host 'https://huggingface.co/mradermacher/Qwen3-4B-Instruct-2507-heretic-uncens-i1-GGUF'
if((Read-Host 'Download this public model? [yes/NO]') -ne 'yes'){exit 0}
New-Item -ItemType Directory -Path $ModelDirectory -Force | Out-Null
$target=Join-Path $ModelDirectory $name
if(Test-Path -LiteralPath $target){throw 'Model already exists; no file was replaced'}
$part=$target+'.part'
& curl.exe --fail --location --retry 3 --proto '=https' --output $part $url
if($LASTEXITCODE -ne 0){throw 'Model download failed; no credential is requested or stored'}
if((Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash -ne $expected){throw 'Model SHA256 does not match the measured snapshot'}
Move-Item -LiteralPath $part -Destination $target
Write-Host "Downloaded: $target"
