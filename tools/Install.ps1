# SPDX-License-Identifier: GPL-3.0-only
[CmdletBinding()]
param(
    [ValidateSet('Auto','AMD')][string]$Backend='Auto',
    [string]$InstallRoot=(Join-Path (Split-Path $PSScriptRoot -Parent) 'runtime'),
    [string]$HipRoot='',
    [switch]$AcceptThirdPartyLicenses,
    [switch]$WithComfyUI,
    [switch]$WithoutComfyUI,
    [switch]$NoDesktopShortcut,
    [switch]$WithSmallGemmPatch,
    [switch]$WithTorchDirectML,
    [string]$HipRuntime='',
    [int]$DxgiDevice=-1,
    [string]$DownloadCache=''
)
$ErrorActionPreference='Stop'
[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
$source=Split-Path $PSScriptRoot -Parent
if(![Environment]::Is64BitOperatingSystem -or ![Environment]::Is64BitProcess){throw 'Use 64-bit Windows PowerShell on Windows x64'}
$components=Get-Content -LiteralPath (Join-Path $source 'profiles\components.json') -Raw | ConvertFrom-Json
$root=[IO.Path]::GetFullPath($InstallRoot)
if($WithComfyUI -and $WithoutComfyUI){throw 'Choose only one ComfyUI option'}
if(!$WithComfyUI -and !$WithoutComfyUI){
    $WithComfyUI=((Read-Host 'Also install ComfyUI and its desktop shortcut? [Y/n]') -notmatch '^(n|no)$')
}
if(!$AcceptThirdPartyLicenses){
    Write-Host 'Read THIRD_PARTY_NOTICES.md and the linked vendor terms before downloading.'
    Write-Host 'No model or third-party executable is distributed in this installer archive.'
    if((Read-Host 'Accept the applicable third-party terms and continue? [yes/NO]') -ne 'yes'){throw 'Cancelled before download'}
}
if(Test-Path -LiteralPath $root){throw 'Choose a new installation folder. Existing installations are never overwritten.'}
if($Backend -eq 'Auto'){
    $names=@(Get-CimInstance Win32_VideoController | ForEach-Object {$_.Name}) -join ' '
    if($names -match 'RX\s*(470|480|570|580|590)'){$Backend='AMD'}
    else{throw 'This installer targets AMD Polaris/gfx803. For a compatible card not recognized by name, select -Backend AMD explicitly.'}
}
if($HipRuntime -and !(Test-Path -LiteralPath $HipRuntime -PathType Leaf)){throw 'The explicit existing HIP runtime does not exist'}
if(!$HipRuntime -and !(Test-Path -LiteralPath "$env:SystemRoot\System32\amdhip64.dll") -and !($HipRoot -and (Test-Path -LiteralPath (Join-Path $HipRoot 'bin\amdhip64.dll')))){
    throw 'The installed AMD driver does not provide amdhip64.dll. This installer never installs or replaces a GPU driver.'
}
New-Item -ItemType Directory -Path $root -Force | Out-Null
$downloads=Join-Path $root '.downloads'
New-Item -ItemType Directory -Path $downloads -Force | Out-Null
function CheckedRun([string]$Exe,[string[]]$Arguments){
    & $Exe @Arguments
    if($LASTEXITCODE -ne 0){throw "Command failed ($LASTEXITCODE): $([IO.Path]::GetFileName($Exe))"}
}
function Fetch([string]$Name,[string]$FileName){
    $asset=$components.$Name
    Write-Host "Fetching pinned component: $Name"
    $path=Join-Path $downloads $FileName
    $fromCache=$false
    if($DownloadCache -and (Test-Path -LiteralPath (Join-Path $DownloadCache $FileName) -PathType Leaf)){
        $path=Join-Path $DownloadCache $FileName
        $fromCache=$true
    }
    $part=$path+'.part'
    $candidate=$path
    if(!(Test-Path -LiteralPath $path)){
        CheckedRun 'curl.exe' @('--silent','--show-error','--fail','--location','--retry','3','--proto','=https','--output',$part,[string]$asset.url)
        $candidate=$part
    }
    if($asset.sha256){
        if((Get-FileHash -LiteralPath $candidate -Algorithm SHA256).Hash -ne $asset.sha256){throw "SHA256 mismatch: $Name"}
    } elseif($asset.signature_publisher){
        $signature=Get-AuthenticodeSignature -LiteralPath $candidate
        if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch $asset.signature_publisher){throw "Publisher signature verification failed: $Name"}
    } else{throw "No integrity policy for $Name"}
    if($candidate -eq $part -and !$fromCache){Move-Item -LiteralPath $part -Destination $path}
    return $path
}
function AdminExtract([string]$Msi,[string]$Destination){
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $process=Start-Process -FilePath "$env:SystemRoot\System32\msiexec.exe" -ArgumentList @('/a',('"'+$Msi+'"'),'/qn',('TARGETDIR="'+$Destination+'"')) -WindowStyle Hidden -Wait -PassThru
    if($process.ExitCode -ne 0){throw "MSI administrative extraction failed ($($process.ExitCode)). No driver installation was requested."}
}
if(@('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','vcruntime140_1.dll') | Where-Object {!(Test-Path -LiteralPath (Join-Path "$env:SystemRoot\System32" $_))}){
    $vc=Fetch 'vcredist' 'vc_redist.x64.exe'
    Write-Host 'Installing signed Microsoft VC++ x64 runtime; Windows may request administrator consent.'
    $process=Start-Process -FilePath $vc -ArgumentList @('/install','/quiet','/norestart') -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    if($process.ExitCode -notin @(0,1638,3010)){throw "VC++ installation failed ($($process.ExitCode))"}
    if(@('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','vcruntime140_1.dll') | Where-Object {!(Test-Path -LiteralPath (Join-Path "$env:SystemRoot\System32" $_))}){throw 'VC++ runtime remains unavailable; restart Windows if requested by Microsoft, then retry in a fresh folder'}
}
$pythonArchive=Fetch 'python' 'python-embed.zip'
$pipWheel=Fetch 'pip' 'pip-bootstrap.whl'
$cuda=Join-Path $root 'envs\cuda'
$onnx=Join-Path $root 'envs\onnx'
$diagnostics=Join-Path $root 'envs\diagnostics'
New-Item -ItemType Directory -Path $diagnostics -Force | Out-Null
Expand-Archive -LiteralPath $pythonArchive -DestinationPath $diagnostics
@('python312.zip','.','import site') | Set-Content -LiteralPath (Join-Path $diagnostics 'python312._pth') -Encoding ASCII
$probePython=Join-Path $diagnostics 'python.exe'
$selection=& $probePython -I (Join-Path $source 'tools\gpu_devices.py') --select --index $DxgiDevice
if($LASTEXITCODE){throw 'Choose a physical Radeon with -DxgiDevice; see the listed DXGI candidates'}
$device=$selection | ConvertFrom-Json
foreach($environment in @($cuda,$onnx)){
    New-Item -ItemType Directory -Path $environment -Force | Out-Null
    Expand-Archive -LiteralPath $pythonArchive -DestinationPath $environment
    $site=Join-Path $environment 'Lib\site-packages'
    New-Item -ItemType Directory -Path $site -Force | Out-Null
    @('python312.zip','.','Lib','Lib\site-packages','import site') | Set-Content -LiteralPath (Join-Path $environment 'python312._pth') -Encoding ASCII
    CheckedRun (Join-Path $environment 'python.exe') @('-I',(Join-Path $source 'tools\prepare.py'),'unzip',$pipWheel,$site)
}
$cudaPython=Join-Path $cuda 'python.exe'
$onnxPython=Join-Path $onnx 'python.exe'
$cudaRequirements=Join-Path $source 'requirements\cuda-py312.txt'
$installRequirements=if($WithComfyUI){Join-Path $source 'requirements\comfy-py312.txt'}else{$cudaRequirements}
$onnxRequirements=Join-Path $source 'requirements\onnx-py312.txt'
$torchWheel=Fetch 'torch' 'torch-2.2.1+cu118-cp312-cp312-win_amd64.whl'
CheckedRun $cudaPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--disable-pip-version-check','-c',$cudaRequirements,$torchWheel,'torchvision==0.17.1+cu118','torchaudio==2.2.1+cu118','--index-url','https://download.pytorch.org/whl/cu118','--extra-index-url','https://pypi.org/simple')
CheckedRun $cudaPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--only-binary=:all:','--disable-pip-version-check','-r',$installRequirements,'--index-url','https://pypi.org/simple')
CheckedRun $onnxPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--only-binary=:all:','--disable-pip-version-check','-r',$onnxRequirements,'--index-url','https://pypi.org/simple')
if($WithTorchDirectML){
    $dml=Join-Path $root 'envs\torch-directml'
    New-Item -ItemType Directory -Path $dml -Force | Out-Null
    $python311=Fetch 'python311' 'python-3.11.9-embed-amd64.zip'
    Expand-Archive -LiteralPath $python311 -DestinationPath $dml
    $site=Join-Path $dml 'Lib\site-packages'
    New-Item -ItemType Directory -Path $site -Force | Out-Null
    @('python311.zip','.','Lib','Lib\site-packages','import site') | Set-Content -LiteralPath (Join-Path $dml 'python311._pth') -Encoding ASCII
    $dmlPython=Join-Path $dml 'python.exe'
    CheckedRun $dmlPython @('-I',(Join-Path $source 'tools\prepare.py'),'unzip',$pipWheel,$site)
    $dmlTorch=Fetch 'torch_dml_cpu' 'torch-2.4.1+cpu-cp311-cp311-win_amd64.whl'
    $dmlVision=Fetch 'torch_dml_vision' 'torchvision-0.19.1+cpu-cp311-cp311-win_amd64.whl'
    $dmlProvider=Fetch 'torch_dml' 'torch_directml-0.2.5.dev240914-cp311-cp311-win_amd64.whl'
    CheckedRun $dmlPython @('-I','-m','pip','--isolated','install','--no-deps','--no-cache-dir',$dmlTorch,$dmlVision,$dmlProvider)
    CheckedRun $dmlPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--require-hashes','-r',(Join-Path $source 'requirements\torch-directml-py311.txt'),'--index-url','https://pypi.org/simple')
}
$apps=Join-Path $root 'apps'
New-Item -ItemType Directory -Path $apps -Force | Out-Null
if($WithComfyUI){
$comfyArchive=Fetch 'comfy' 'comfy-source.zip'
$unpacked=Join-Path $downloads 'comfy-source'
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'unzip',$comfyArchive,$unpacked)
$sourceDirectory=Get-ChildItem -LiteralPath $unpacked -Directory | Select-Object -First 1
$comfy=Join-Path $apps 'ComfyUI'
Move-Item -LiteralPath $sourceDirectory.FullName -Destination $comfy
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'patch-comfy',$comfy,(Join-Path $source 'patches\comfy-offline-startup.diff'))
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'prune-comfy',$comfy)
}
if($Backend -eq 'AMD'){
    $sevenzip=Get-Command '7z.exe' -ErrorAction SilentlyContinue
    $sevenzipPath=if($sevenzip){$sevenzip.Source}else{Join-Path $env:ProgramFiles '7-Zip\7z.exe'}
    if(!(Test-Path -LiteralPath $sevenzipPath)){
        $msi=Fetch 'sevenzip' '7zip.msi'
        $sevenzipImage=Join-Path $downloads '7zip-admin-image'
        AdminExtract $msi $sevenzipImage
        $found=Get-ChildItem -LiteralPath $sevenzipImage -Recurse -File -Filter '7z.exe' | Select-Object -First 1
        if(!$found){throw '7-Zip extraction did not produce 7z.exe'}
        $sevenzipPath=$found.FullName
    }
    if(!$HipRoot){
        $sdk=Fetch 'hip_sdk' 'hip-sdk-5.7.1.exe'
        $sdkPackages=Join-Path $downloads 'sdk-packages'
        CheckedRun $sevenzipPath @('x','-y',('-o'+$sdkPackages),$sdk,'Packages\Apps\ROCmSDKPackages\*','-r')
        $sdkOnly=Join-Path $sdkPackages 'Packages\Apps\ROCmSDKPackages'
        if(!(Test-Path -LiteralPath $sdkOnly)){throw 'Pinned SDK-only package subtree was not found; no executable installer will be run'}
        $image=Join-Path $downloads 'sdk-admin-image'
        $packages=@(Get-ChildItem -LiteralPath $sdkOnly -Recurse -File -Filter '*.msi')
        if(!$packages.Count){throw 'No SDK MSI packages found'}
        foreach($package in $packages){AdminExtract $package.FullName $image}
        $compiler=Get-ChildItem -LiteralPath $image -Recurse -File -Filter 'amd_comgr0507.dll' | Where-Object {$_.Directory.Name -eq 'bin'} | Select-Object -First 1
        if(!$compiler){throw 'HIP 5.7 compiler library was not extracted'}
        $HipRoot=$compiler.Directory.Parent.FullName
    }
    if(!(Test-Path -LiteralPath (Join-Path $HipRoot 'bin\amd_comgr0507.dll'))){throw 'HIP 5.7.1 SDK root is required'}
    $hipSource=[IO.Path]::GetFullPath($HipRoot).TrimEnd('\')
    if($root.StartsWith($hipSource+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Installation root must not be inside the supplied SDK root'}
    $hip=Join-Path $root 'hip'
    New-Item -ItemType Directory -Path $hip -Force | Out-Null
    Get-ChildItem -LiteralPath $HipRoot -Force | Copy-Item -Destination $hip -Recurse
    $rocArchive=Fetch 'rocblas' 'gfx803-rocblas.7z'
    $rocUnpacked=Join-Path $downloads 'gfx803-rocblas'
    CheckedRun $sevenzipPath @('x','-y',('-o'+$rocUnpacked),$rocArchive)
    $rocDll=Join-Path $rocUnpacked 'rocblas.dll'
    if((Get-FileHash -LiteralPath $rocDll -Algorithm SHA256).Hash -ne '9E88918E65B81EEB128C69CD7A96AB495B101DAF51F6F78F943445D7D4C5F70D'){throw 'Unexpected patched rocBLAS'}
    $library=Join-Path $hip 'bin\rocblas\library'
    # Only a fresh installation-owned SDK copy is modified, never -HipRoot itself.
    if(Test-Path -LiteralPath $library){
        $full=(Resolve-Path -LiteralPath $library).ProviderPath
        if(!$full.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'SDK destination escaped root'}
        if(@(Get-ChildItem -LiteralPath $full -Force -Recurse | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){throw 'Reparse point in SDK copy'}
        Remove-Item -LiteralPath $full -Recurse -Force
    }
    Copy-Item -LiteralPath (Join-Path $rocUnpacked 'library') -Destination $library -Recurse
    Copy-Item -LiteralPath $rocDll -Destination (Join-Path $hip 'bin\rocblas.dll') -Force
    $zludaArchive=Fetch 'zluda' 'zluda-3.9.5.zip'
    CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'unzip',$zludaArchive,$root)
    $torchLib=Join-Path $cuda 'Lib\site-packages\torch\lib'
    $originalNvrtcHash=(Get-FileHash -LiteralPath (Join-Path $torchLib 'nvrtc64_112_0.dll') -Algorithm SHA256).Hash.ToLowerInvariant()
    Copy-Item -LiteralPath (Join-Path $torchLib 'nvrtc64_112_0.dll') -Destination (Join-Path $torchLib 'nvrtc_cuda.dll')
    foreach($pair in @(@('cublas.dll','cublas64_11.dll'),@('cusparse.dll','cusparse64_11.dll'),@('nvrtc.dll','nvrtc64_112_0.dll'))){
        Copy-Item -LiteralPath (Join-Path $root ('zluda\'+$pair[0])) -Destination (Join-Path $torchLib $pair[1]) -Force
    }
}
Copy-Item -LiteralPath (Join-Path $source 'tools') -Destination (Join-Path $root 'tools') -Recurse
Copy-Item -LiteralPath (Join-Path $source 'patches') -Destination (Join-Path $root 'patches') -Recurse
foreach($name in @('Python-CUDA.cmd','Python-ONNX.cmd','Run-ZLUDA.cmd','HIP-Shell.cmd','Check-Zluda.cmd')){Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $root $name)}
if($WithTorchDirectML){Copy-Item -LiteralPath (Join-Path $source 'Python-DirectML.cmd') -Destination $root}
$profile=@{backend=$Backend;version='0.3.0';comfy_installed=[bool]$WithComfyUI;torch_directml_installed=[bool]$WithTorchDirectML;architecture='Polaris/gfx803';comfy_commit='c0c011dca91b595091a4b032770da366fc90fb2b';models_included=$false;dxgi_device_id=$device.dxgi_index}
$profile.original_nvrtc_sha256=$originalNvrtcHash
if($HipRuntime){$profile.hip_runtime=(Resolve-Path -LiteralPath $HipRuntime).ProviderPath}
$profile | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'profile.json') -Encoding UTF8
# PE/version diagnostics only: no device allocation or numerical workload.
CheckedRun $probePython @('-I',(Join-Path $source 'tools\hip_runtime.py'),'--root',$root)
if($WithSmallGemmPatch){
    $build=Join-Path $source ('work\small-gemm-'+[guid]::NewGuid().ToString('N'))
    & (Join-Path $source 'patches\small-gemm\Build-SmallGemm.ps1') -HipRoot (Join-Path $root 'hip') -BackendDll (Join-Path $root 'hip\bin\rocblas.dll') -Python $probePython -OutDir $build
    & (Join-Path $source 'tools\Small-Gemm.ps1') -RuntimeRoot $root -BuildDirectory $build
}
New-Item -ItemType Directory -Path (Join-Path $root 'models'),(Join-Path $root 'temp') -Force | Out-Null
if($WithComfyUI){
    Copy-Item -LiteralPath (Join-Path $source 'Start-ComfyUI.cmd') -Destination $root
    if(!$NoDesktopShortcut){
        $desktop=[Environment]::GetFolderPath('DesktopDirectory')
        if(!$desktop){throw 'Windows did not provide a desktop folder; use -NoDesktopShortcut'}
        $shell=New-Object -ComObject WScript.Shell
        $shortcut=$shell.CreateShortcut((Join-Path $desktop 'ComfyUI - Polaris ZLUDA.lnk'))
        $shortcut.TargetPath=Join-Path $root 'Start-ComfyUI.cmd'
        $shortcut.WorkingDirectory=$root
        $shortcut.Description='ComfyUI with the local Polaris gfx803 ZLUDA runtime'
        $shortcut.Save()
    }
}
Write-Host "Installed into $root. No GPU driver, inference, generation or benchmark was run."
Write-Host 'Download your own licensed models from Hugging Face; see README.md.'
$downloadFull=(Resolve-Path -LiteralPath $downloads).ProviderPath
if(!$downloadFull.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Download cleanup escaped installation root'}
if(@(Get-ChildItem -LiteralPath $downloadFull -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){
    Write-Host 'Download cache contains a reparse point; it has been preserved for manual review.'
}else{Remove-Item -LiteralPath $downloadFull -Recurse -Force}
