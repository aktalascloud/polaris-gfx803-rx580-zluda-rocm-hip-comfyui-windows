# Source-only build helper. Does not install files or modify drivers.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$HipRoot,
    [Parameter(Mandatory=$true)][string]$BackendDll,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$OutDir
)
$ErrorActionPreference = 'Stop'
$hip = (Resolve-Path -LiteralPath $HipRoot).ProviderPath.TrimEnd('\')
$backend = (Resolve-Path -LiteralPath $BackendDll).ProviderPath
$pythonFile = (Resolve-Path -LiteralPath $Python).ProviderPath
$build = [IO.Path]::GetFullPath($OutDir)
if($build -match '[^\x00-\x7f]'){throw 'Small-GEMM native loader requires an ASCII output/runtime path; spaces are supported'}
if (Test-Path -LiteralPath $build) { throw 'Choose a new, nonexistent build directory' }
if ($build.StartsWith($hip+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Build outside the SDK/runtime directory' }
$runtimeParent=Split-Path $hip -Parent
if((Test-Path -LiteralPath (Join-Path $runtimeParent 'profile.json')) -or (Test-Path -LiteralPath (Join-Path $runtimeParent 'installation-manifest.json'))){
    if($build.StartsWith($runtimeParent.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Build outside the complete private runtime, not only outside hip/bin'}
}
if ((Get-FileHash -LiteralPath $backend -Algorithm SHA256).Hash -ne '9E88918E65B81EEB128C69CD7A96AB495B101DAF51F6F78F943445D7D4C5F70D') { throw 'Unsupported backend; supply original advanced-lvl-up DLL, never the shim' }
$bin = Join-Path $hip 'bin'
foreach ($name in @('clang.exe','clang-offload-bundler.exe','lld-link.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $bin $name))) { throw ('Missing SDK build tool: '+$name) }
}
$imports = Join-Path $build 'imports'
New-Item -ItemType Directory -Path $imports -Force | Out-Null
& $pythonFile (Join-Path $PSScriptRoot 'make-exports.py') --backend $backend --output (Join-Path $build 'rocblas.def')
if ($LASTEXITCODE -ne 0) { throw 'Export generation failed' }
Copy-Item -LiteralPath $backend -Destination (Join-Path $build 'rocblas_backend.dll')
& (Join-Path $bin 'clang.exe') -x hip --offload-arch=gfx803 --cuda-device-only -nogpuinc -nogpulib -O2 -fno-fast-math -c (Join-Path $PSScriptRoot 'kernel.hip') -o (Join-Path $build 'kernel.o')
if ($LASTEXITCODE -ne 0) { throw 'HIP kernel compilation failed' }
& (Join-Path $bin 'clang-offload-bundler.exe') -unbundle -type=o -targets=hipv4-amdgcn-amd-amdhsa--gfx803 ('-input='+(Join-Path $build 'kernel.o')) ('-output='+(Join-Path $build 'rocblas_small_gemm.hsaco'))
if ($LASTEXITCODE -ne 0) { throw 'HIP code-object extraction failed' }
& (Join-Path $bin 'lld-link.exe') /dll /noentry /machine:x64 ('/def:'+(Join-Path $PSScriptRoot 'kernel32.def')) ('/out:'+(Join-Path $imports 'kernel32.dll')) ('/implib:'+(Join-Path $imports 'kernel32.lib'))
if ($LASTEXITCODE -ne 0) { throw 'Import-library generation failed' }
& (Join-Path $bin 'clang.exe') --target=x86_64-pc-windows-msvc -O2 -fno-stack-protector -fno-builtin -c (Join-Path $PSScriptRoot 'shim.c') -o (Join-Path $build 'shim.obj')
if ($LASTEXITCODE -ne 0) { throw 'Shim compilation failed' }
& (Join-Path $bin 'lld-link.exe') /dll /noentry /nodefaultlib /machine:x64 ('/def:'+(Join-Path $build 'rocblas.def')) ('/out:'+(Join-Path $build 'rocblas.dll')) (Join-Path $build 'shim.obj') (Join-Path $imports 'kernel32.lib')
if ($LASTEXITCODE -ne 0) { throw 'Shim link failed' }
& $pythonFile -I (Join-Path $PSScriptRoot 'make-exports.py') --backend $backend --repair-shim (Join-Path $build 'rocblas.dll')
if ($LASTEXITCODE -ne 0) { throw 'Forwarder ordinal repair failed' }
& $pythonFile -I (Join-Path $PSScriptRoot 'make-exports.py') --backend $backend --verify-shim (Join-Path $build 'rocblas.dll')
if ($LASTEXITCODE -ne 0) { throw 'Export/ordinal verification failed' }
$hashes=@{}
foreach($file in @('rocblas.dll','rocblas_backend.dll','rocblas_small_gemm.hsaco')){$hashes[$file]=(Get-FileHash -LiteralPath (Join-Path $build $file) -Algorithm SHA256).Hash.ToLowerInvariant()}
@{version=1;files=$hashes;target='gfx803';max_dimension=8;dtype='FP32';context_scope='initial HIP context, device 0'} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $build 'build.json') -Encoding UTF8
Write-Output ('Build ready, nothing installed: '+$build)
