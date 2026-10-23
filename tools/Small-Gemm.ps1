# SPDX-License-Identifier: GPL-3.0-only
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$RuntimeRoot,
    [ValidateSet('Install','Rollback')][string]$Action='Install',
    [string]$BuildDirectory=''
)
$ErrorActionPreference='Stop'
$expected='9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d'
$root=(Resolve-Path -LiteralPath $RuntimeRoot).ProviderPath.TrimEnd('\')
if($root -match '[^\x00-\x7f]'){throw 'Small-GEMM requires an ASCII runtime path; spaces are supported'}
if(!(Test-Path -LiteralPath (Join-Path $root 'profile.json'))){throw 'Select an installed private toolkit runtime with profile.json'}
foreach($directory in @($root,(Join-Path $root 'hip'),(Join-Path $root 'hip\bin'))){
    if((Get-Item -LiteralPath $directory -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse runtime path refused'}
}
$bin=Join-Path $root 'hip\bin'
$dll=Join-Path $bin 'rocblas.dll'
$backend=Join-Path $bin 'rocblas_backend.dll'
$kernel=Join-Path $bin 'rocblas_small_gemm.hsaco'
$metadata=Join-Path $bin 'small-gemm-patch.json'
function Hash([string]$path){return (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
function AssertClosed {
    foreach($path in @($dll,$backend,$kernel,$metadata)){
        if(Test-Path -LiteralPath $path){
            if((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse artifact refused'}
            try{$stream=[IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None);$stream.Dispose()}
            catch{throw 'Close all consumers of this runtime before changing its DLLs. No automatic rollback is attempted.'}
        }
    }
}
AssertClosed
if($Action -eq 'Rollback'){
    if(!(Test-Path -LiteralPath $metadata)){if((Hash $dll) -eq $expected){Write-Output 'Already original backend';return};throw 'No trusted patch receipt'}
    $receipt=Get-Content -LiteralPath $metadata -Raw | ConvertFrom-Json
    if((Hash $backend) -ne $expected){throw 'Original backend backup hash mismatch'}
    if((Hash $dll) -eq $expected){Write-Output 'Already rolled back';return}
    if((Hash $dll) -ne $receipt.files.'rocblas.dll'){throw 'Installed DLL changed; rollback refused'}
    AssertClosed
    $temporary=Join-Path $bin ('rocblas-restore-'+[guid]::NewGuid().ToString('N')+'.tmp')
    Copy-Item -LiteralPath $backend -Destination $temporary
    [IO.File]::Replace($temporary,$dll,[NullString]::Value)
    $receipt | Add-Member -NotePropertyName enabled -NotePropertyValue $false -Force
    $receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadata -Encoding UTF8
    if((Hash $dll) -ne $expected){throw 'Rollback integrity failure'}
    Write-Output 'Original backend restored; kernel library and driver unchanged'
    return
}
$build=(Resolve-Path -LiteralPath $BuildDirectory).ProviderPath
$receipt=Get-Content -LiteralPath (Join-Path $build 'build.json') -Raw | ConvertFrom-Json
foreach($name in @('rocblas.dll','rocblas_backend.dll','rocblas_small_gemm.hsaco')){
    $file=Join-Path $build $name
    if((Get-Item -LiteralPath $file -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Reparse build artifact refused'}
    if((Hash $file) -ne $receipt.files.$name){throw 'Build artifact hash mismatch'}
}
if((Hash (Join-Path $build 'rocblas_backend.dll')) -ne $expected){throw 'Wrong backend'}
$python=Join-Path $root 'envs\diagnostics\python.exe'
& $python -I (Join-Path $root 'patches\small-gemm\make-exports.py') --backend (Join-Path $build 'rocblas_backend.dll') --verify-shim (Join-Path $build 'rocblas.dll')
if($LASTEXITCODE){throw 'Exports/ordinals verification failed'}
if((Hash $dll) -eq $receipt.files.'rocblas.dll'){
    if(!(Test-Path -LiteralPath $metadata) -or (Hash $backend) -ne $expected -or (Hash $kernel) -ne $receipt.files.'rocblas_small_gemm.hsaco'){throw 'Incomplete existing patch'}
    Write-Output 'Identical patch already installed';return
}
if((Hash $dll) -ne $expected){throw 'Only the known original backend can be patched'}
foreach($path in @($backend,$kernel)){
    if(Test-Path -LiteralPath $path){
        $name=[IO.Path]::GetFileName($path)
        if((Hash $path) -ne $receipt.files.$name){throw 'Conflicting existing patch artifact'}
    }
}
AssertClosed
# Stage every new file before the sole atomic replacement of the public DLL.
foreach($name in @('rocblas_backend.dll','rocblas_small_gemm.hsaco')){
    $destination=Join-Path $bin $name
    if(!(Test-Path -LiteralPath $destination)){Copy-Item -LiteralPath (Join-Path $build $name) -Destination $destination}
}
$temporary=Join-Path $bin ('rocblas-install-'+[guid]::NewGuid().ToString('N')+'.tmp')
Copy-Item -LiteralPath (Join-Path $build 'rocblas.dll') -Destination $temporary
$receipt | Add-Member -NotePropertyName enabled -NotePropertyValue $true -Force
$receipt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $metadata -Encoding UTF8
AssertClosed
[IO.File]::Replace($temporary,$dll,[NullString]::Value)
if((Hash $dll) -ne $receipt.files.'rocblas.dll'){throw 'Installed shim hash mismatch'}
Write-Output 'Small-GEMM installed into private runtime; no GPU tests run'
