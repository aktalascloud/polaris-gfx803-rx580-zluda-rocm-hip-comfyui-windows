# SPDX-License-Identifier: GPL-3.0-only
[CmdletBinding()]
param(
    [ValidateSet('Auto','AMD')][string]$Backend='Auto',
    [string]$InstallRoot=(Join-Path (Split-Path $PSScriptRoot -Parent) 'runtime'),
    [string]$HipRoot='',
    [switch]$AcceptThirdPartyLicenses
)
$ErrorActionPreference='Stop'
[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12
$source=Split-Path $PSScriptRoot -Parent
if(![Environment]::Is64BitOperatingSystem -or ![Environment]::Is64BitProcess){throw 'Use 64-bit Windows PowerShell on Windows x64'}
$components=Get-Content -LiteralPath (Join-Path $source 'profiles\components.json') -Raw | ConvertFrom-Json
$root=[IO.Path]::GetFullPath($InstallRoot)
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
if(!(Test-Path -LiteralPath "$env:SystemRoot\System32\amdhip64.dll")){
    throw 'The installed AMD driver does not provide amdhip64.dll. This installer never installs or replaces a GPU driver.'
}
if(@('msvcp140.dll','msvcp140_1.dll','msvcp140_2.dll','vcruntime140_1.dll') | Where-Object {!(Test-Path -LiteralPath (Join-Path "$env:SystemRoot\System32" $_))}){
    throw 'Microsoft Visual C++ 2015-2022 x64 Runtime is required. Download it from https://aka.ms/vs/17/release/vc_redist.x64.exe, then retry.'
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
    if(!(Test-Path -LiteralPath $path)){
        $part=$path+'.part'
        CheckedRun 'curl.exe' @('--silent','--show-error','--fail','--location','--retry','3','--proto','=https','--output',$part,[string]$asset.url)
        Move-Item -LiteralPath $part -Destination $path
    }
    if($asset.sha256){
        if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $asset.sha256){throw "SHA256 mismatch: $Name"}
    } elseif($asset.signature_publisher){
        $signature=Get-AuthenticodeSignature -LiteralPath $path
        if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch $asset.signature_publisher){throw 'AMD installer signature verification failed'}
    } else{throw "No integrity policy for $Name"}
    return $path
}
function AdminExtract([string]$Msi,[string]$Destination){
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $process=Start-Process -FilePath "$env:SystemRoot\System32\msiexec.exe" -ArgumentList @('/a',('"'+$Msi+'"'),'/qn',('TARGETDIR="'+$Destination+'"')) -WindowStyle Hidden -Wait -PassThru
    if($process.ExitCode -ne 0){throw "MSI administrative extraction failed ($($process.ExitCode)). No driver installation was requested."}
}
$pythonArchive=Fetch 'python' 'python-embed.zip'
$pipWheel=Fetch 'pip' 'pip-bootstrap.whl'
$cuda=Join-Path $root 'envs\cuda'
$onnx=Join-Path $root 'envs\onnx'
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
$onnxRequirements=Join-Path $source 'requirements\onnx-py312.txt'
$torchWheel=Fetch 'torch' 'torch-2.2.1+cu118-cp312-cp312-win_amd64.whl'
CheckedRun $cudaPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--disable-pip-version-check','-c',$cudaRequirements,$torchWheel,'torchvision==0.17.1+cu118','torchaudio==2.2.1+cu118','--index-url','https://download.pytorch.org/whl/cu118','--extra-index-url','https://pypi.org/simple')
CheckedRun $cudaPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--only-binary=:all:','--disable-pip-version-check','-r',$cudaRequirements,'--index-url','https://pypi.org/simple')
CheckedRun $onnxPython @('-I','-m','pip','--isolated','install','--no-cache-dir','--only-binary=:all:','--disable-pip-version-check','-r',$onnxRequirements,'--index-url','https://pypi.org/simple')
$apps=Join-Path $root 'apps'
New-Item -ItemType Directory -Path $apps -Force | Out-Null
$comfyArchive=Fetch 'comfy' 'comfy-source.zip'
$unpacked=Join-Path $downloads 'comfy-source'
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'unzip',$comfyArchive,$unpacked)
$sourceDirectory=Get-ChildItem -LiteralPath $unpacked -Directory | Select-Object -First 1
$comfy=Join-Path $apps 'ComfyUI'
Move-Item -LiteralPath $sourceDirectory.FullName -Destination $comfy
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'patch-comfy',$comfy,(Join-Path $source 'patches\comfy-offline-startup.diff'))
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'prune-comfy',$comfy)
$kobold=Join-Path $apps 'KoboldCPP'
New-Item -ItemType Directory -Path $kobold -Force | Out-Null
$koboldArchive=Fetch 'kobold' 'koboldcpp-oldpc.exe'
CheckedRun $cudaPython @('-I',(Join-Path $source 'tools\prepare.py'),'extract-kobold',$koboldArchive,$kobold)
$frontend=Fetch 'kobold_python' 'koboldcpp.py'
Copy-Item -LiteralPath $frontend -Destination (Join-Path $kobold 'koboldcpp.py')
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
    Copy-Item -LiteralPath (Join-Path $torchLib 'nvrtc64_112_0.dll') -Destination (Join-Path $torchLib 'nvrtc_cuda.dll')
    foreach($pair in @(@('cublas.dll','cublas64_11.dll'),@('cusparse.dll','cusparse64_11.dll'),@('nvrtc.dll','nvrtc64_112_0.dll'))){
        Copy-Item -LiteralPath (Join-Path $root ('zluda\'+$pair[0])) -Destination (Join-Path $torchLib $pair[1]) -Force
    }
    Copy-Item -LiteralPath (Join-Path $root 'zluda\cublas.dll') -Destination (Join-Path $kobold 'cublas64_11.dll') -Force
}
Copy-Item -LiteralPath (Join-Path $source 'tools') -Destination (Join-Path $root 'tools') -Recurse
foreach($name in @('Start-ComfyUI.cmd','Start-Kobold.cmd','Python-CUDA.cmd','Python-ONNX.cmd','Download-Model.cmd')){Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $root $name)}
@{backend=$Backend;version='0.1.1';architecture='Polaris/gfx803';comfy_commit='c0c011dca91b595091a4b032770da366fc90fb2b';models_included=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'profile.json') -Encoding UTF8
New-Item -ItemType Directory -Path (Join-Path $root 'models'),(Join-Path $root 'temp') -Force | Out-Null
Write-Host "Installed into $root. No GPU driver, inference, generation or benchmark was run."
Write-Host 'Download your own licensed models from Hugging Face; see README.md.'
$downloadFull=(Resolve-Path -LiteralPath $downloads).ProviderPath
if(!$downloadFull.StartsWith($root.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Download cleanup escaped installation root'}
if(@(Get-ChildItem -LiteralPath $downloadFull -Recurse -Force | Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){
    Write-Host 'Download cache contains a reparse point; it has been preserved for manual review.'
}else{Remove-Item -LiteralPath $downloadFull -Recurse -Force}
