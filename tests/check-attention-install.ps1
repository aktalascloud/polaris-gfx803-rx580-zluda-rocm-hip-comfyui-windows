# SPDX-License-Identifier: GPL-3.0-only
# Exercise the actual installer ComfyUI block on pinned source, not an ML runtime.
param([Parameter(Mandatory=$true)][string]$Archive,[Parameter(Mandatory=$true)][string]$Python,[Parameter(Mandatory=$true)][string]$WorkDirectory,[Parameter(Mandatory=$true)][string]$Report)
$ErrorActionPreference='Stop'
$source=Split-Path $PSScriptRoot -Parent
$components=Get-Content -LiteralPath (Join-Path $source 'profiles\components.json') -Raw | ConvertFrom-Json
if((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash -ne $components.comfy.sha256){throw 'Wrong source archive'}
$work=[IO.Path]::GetFullPath($WorkDirectory)
if(Test-Path -LiteralPath $work){throw 'Choose a fresh test folder'}
$downloads=Join-Path $work '.downloads';$apps=Join-Path $work 'apps'
New-Item -ItemType Directory -Path $downloads,$apps -Force | Out-Null
$WithComfyUI=$true;$cudaPython=$Python
function Fetch([string]$Name,[string]$FileName){if($Name -ne 'comfy'){throw 'Unexpected fetch'};return $Archive}
function CheckedRun([string]$Exe,[string[]]$Arguments){& $Exe @Arguments;if($LASTEXITCODE){throw 'Installer ComfyUI step failed'}}
$t=$null;$e=$null;$ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path $source 'tools\Install.ps1'),[ref]$t,[ref]$e)
$blocks=@($ast.EndBlock.Statements | Where-Object {$_ -is [System.Management.Automation.Language.IfStatementAst] -and $_.Extent.Text.Contains("'patch-comfy'")})
if($blocks.Count -ne 1){throw 'Could not isolate the real ComfyUI installation block'}
. ([scriptblock]::Create($blocks[0].Extent.Text))
$receipt=Join-Path $comfy 'comfy\ldm\modules\polaris_attention.json'
$info=Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
if(!$info.enabled){throw 'Default attention hook missing'}
$module=Join-Path $comfy 'comfy\ldm\modules\attention.py'
$before=(Get-FileHash -LiteralPath $module).Hash
CheckedRun $Python @('-I',(Join-Path $source 'tools\attention_patch.py'),'install',$comfy)
if((Get-FileHash -LiteralPath $module).Hash -ne $before){throw 'Repeated apply changed the hook'}
CheckedRun $Python @('-I',(Join-Path $source 'tools\attention_patch.py'),'rollback',$comfy)
$disabled=Get-Content -LiteralPath $receipt -Raw | ConvertFrom-Json
if($disabled.enabled){throw 'Rollback did not disable the hook'}
CheckedRun $Python @('-I',(Join-Path $source 'tools\attention_patch.py'),'install',$comfy)
if((Get-FileHash -LiteralPath $module).Hash -ne $before){throw 'Reapply changed the hook'}
$result=@{actual_installer_comfy_block=$true;default_enabled=$true;idempotent=$true;rollback_verified=$true;reapply_verified=$true;models_used=$false;gpu_workload_run=$false;scope='ComfyUI source installation block only; full dependency/bootstrap installer not repeated'}
$result | ConvertTo-Json | Set-Content -LiteralPath $Report -Encoding UTF8
$result | ConvertTo-Json -Compress
