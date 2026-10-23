# SPDX-License-Identifier: GPL-3.0-only
# Exercise the real Fetch function with a synthetic downloader, never network/GPU.
param([Parameter(Mandatory=$true)][string]$WorkDirectory)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($WorkDirectory)
if(Test-Path -LiteralPath $root){throw 'Choose a new test directory'}
New-Item -ItemType Directory -Path $root -Force | Out-Null
$downloads=Join-Path $root 'downloads';$DownloadCache=Join-Path $root 'cache'
New-Item -ItemType Directory -Path $downloads,$DownloadCache -Force | Out-Null
$payload=[Text.Encoding]::UTF8.GetBytes('synthetic-package')
$sourceFile=Join-Path $root 'payload';[IO.File]::WriteAllBytes($sourceFile,$payload)
$expected=(Get-FileHash -LiteralPath $sourceFile -Algorithm SHA256).Hash
$components=[pscustomobject]@{fixture=[pscustomobject]@{url='https://publisher.invalid/artifact';sha256=$expected}}
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile((Join-Path (Split-Path $PSScriptRoot -Parent) 'tools\Install.ps1'),[ref]$tokens,[ref]$errors)
$definition=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Fetch'},$true)
if(!$definition){throw 'Fetch function not found'}
. ([scriptblock]::Create($definition.Extent.Text))
$script:requests=0;$script:response=$payload
function CheckedRun([string]$Exe,[string[]]$Arguments){
    if($Exe -ne 'curl.exe'){throw 'Unexpected downloader'}
    $script:requests++
    $position=[array]::IndexOf($Arguments,'--output')
    [IO.File]::WriteAllBytes($Arguments[$position+1],$script:response)
}
$results=@()
# Valid readonly cache is hash-checked and returned without downloading.
$cacheFile=Join-Path $DownloadCache 'valid.bin';[IO.File]::WriteAllBytes($cacheFile,$payload)
if((Fetch fixture 'valid.bin') -ne $cacheFile -or $script:requests){throw 'Valid cache was not reused'}
$results+='verified_cache_reused'
# Existing corrupt cache never counts as a verified component.
[IO.File]::WriteAllBytes((Join-Path $DownloadCache 'bad.bin'),[byte[]]@(1,2,3))
$rejected=$false
try{Fetch fixture 'bad.bin' | Out-Null}catch{if($_ -notmatch 'SHA256 mismatch'){throw};$rejected=$true}
if(!$rejected){throw 'Corrupt cache was accepted'}
$results+='corrupt_cache_rejected'
# A .part in the supplied cache is neither complete nor reused.
[IO.File]::WriteAllBytes((Join-Path $DownloadCache 'partial.bin.part'),[byte[]]@(1))
$file=Fetch fixture 'partial.bin'
if($script:requests -ne 1 -or (Get-FileHash -LiteralPath $file).Hash -ne $expected){throw 'A partial cache was accepted'}
$results+='partial_cache_ignored'
# A corrupt fresh transfer stays .part; the ready filename is never published.
$script:response=[byte[]]@(3,2,1);$rejected=$false
try{Fetch fixture 'transfer.bin' | Out-Null}catch{if($_ -notmatch 'SHA256 mismatch'){throw};$rejected=$true}
if(!$rejected -or (Test-Path -LiteralPath (Join-Path $downloads 'transfer.bin'))){throw 'Bad transfer became ready'}
$script:response=$payload
$file=Fetch fixture 'transfer.bin'
if((Get-FileHash -LiteralPath $file).Hash -ne $expected){throw 'Retry failed'}
$results+='invalid_transfer_not_promoted_and_retry_verified'
@{passed=$true;cases=$results;network_used=$false;gpu_used=$false} | ConvertTo-Json -Compress
