# Licenses and publication boundary

Project scripts, portable adaptations and the ComfyUI-derived patch are
**GPL-3.0-only**; see [LICENSE](LICENSE). Existing upstream copyright notices remain
applicable. The small-GEMM contribution is covered by [its notice](patches/small-gemm/NOTICE.txt).

**Only source and sanitized numeric evidence are published.** Vendor DLLs/EXEs,
wheels, SDK/driver installers, generated code objects, models, raw logs, private
configuration, credentials and caches are not redistributed. Setup downloads
components from their publishers; their terms require acceptance and are not
replaced by the project's GPL license.

| Component | Terms / source |
|---|---|
| Python 3.12.10 / 3.11.9 | [Python license](https://docs.python.org/3/license.html), [distributions](https://www.python.org/ftp/python/) |
| pip 25.2 | [MIT and bundled dependency notices](https://github.com/pypa/pip/blob/25.2/LICENSE.txt) |
| Torch 2.2.1 / 2.4.1 | [Project license](https://github.com/pytorch/pytorch/blob/v2.2.1/LICENSE); bundled vendor terms also apply |
| torchvision / torchaudio | [Vision source](https://github.com/pytorch/vision/tree/v0.17.1), [audio source](https://github.com/pytorch/audio/tree/v2.2.1), respective BSD-style notices |
| CUDA / NVRTC / cuDNN | [CUDA 11.8 EULA](https://docs.nvidia.com/cuda/archive/11.8.0/eula/index.html) and applicable component terms |
| ComfyUI-Zluda | [Pinned GPL source](https://github.com/patientx/ComfyUI-Zluda/tree/c0c011dca91b595091a4b032770da366fc90fb2b) |
| ZLUDA fork | [Pinned MIT / Apache-2.0 source](https://github.com/lshqqytiger/ZLUDA/tree/5e717459179dc272b7d7d23391f0fad66c7459cf) |
| Patched gfx803 rocBLAS | [Publisher/source release](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/releases/tag/v1.0.0), [upstream rocBLAS license](https://github.com/ROCm/rocBLAS/blob/rocm-5.7.1/LICENSE.md); patch binary is not mirrored |
| HIP SDK 5.7.1 | [AMD terms](https://www.amd.com/en/developer/resources/rocm-hub/eula/licenses.html), [SDK page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) |
| 7-Zip | [License: LGPL and applicable unRAR restrictions](https://www.7-zip.org/license.txt), [source/downloads](https://www.7-zip.org/download.html) |
| ONNX Runtime DirectML | [MIT/component notices](https://github.com/microsoft/onnxruntime/blob/v1.24.4/LICENSE) |
| Torch DirectML | [Microsoft source/component notices](https://github.com/microsoft/DirectML/tree/master/PyTorch); downloaded wheel retains its licenses |
| OpenVINO | [Apache-2.0/component notices](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE) |
| VC++ x64 runtime | [Microsoft download](https://aka.ms/vs/17/release/vc_redist.x64.exe); installed under Microsoft's terms if missing |

## Vendor limits

AMD [deployment guidance](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/deployment-guidelines.html)
uses the installed driver's HIP runtime. This toolkit does not mirror driver DLLs
or overwrite Windows/system drivers. SDK-only extraction does not run AMD Setup
or its driver packages; an explicit existing local HIP runtime is not a new driver.

CUDA 11.8 EULA §1.2(8) restricts translating SDK-generated artifacts to non-NVIDIA
targets. An open-source installer does not remove those restrictions or establish
every recipient's rights; users must obtain the rights their third-party use needs.
No NVIDIA binaries or vendor endorsement are supplied by this project.

The patched rocBLAS publisher lacks an explicit project license; upstream rocBLAS
licensing alone does not establish permission for all third-party modifications.
Its binaries are therefore downloaded from that publisher rather than mirrored.
Small-GEMM is local Windows source inspired by Schaka's finding, not a distribution
of Linux ROCm binaries or a claim of a complete port.

## Privacy

Published results exclude personal prompts, host paths, serials, profiles, keys,
authentication files and raw logs. Setup configures no sign-in or stored API key.
Normal publisher downloads and local application interfaces do not grant cloud access.

The bounded attention adaptation is GPL-3.0-only, inspired by advanced-lvl-up's GPL ComfyUI helper. Attribution/source and the rewritten contract are in [ATTENTION.md](docs/ATTENTION.md). No upstream binary or complete third-party attention module is repackaged.
