# Third-party software, licenses and distribution boundary

Project-authored scripts and the ComfyUI-derived patch are distributed under
**GPL-3.0-only**, with the full GPL text in [LICENSE](LICENSE).
Existing upstream copyright notices in the patch remain applicable.

**The repository and release do not redistribute third-party executables,
DLLs, wheels, GPU drivers, model weights, SDK installers, caches or logs.**
The source installer downloads upstream artifacts on the recipient's machine.
Acceptance of the applicable upstream terms is required before setup proceeds.
This project's GPL license does not relicense those downloads or grant rights
that their publishers have not granted.

| Download / referenced project | Upstream terms / source |
|---|---|
| Python 3.12.10 embeddable | [Python license](https://docs.python.org/3/license.html); [official distribution](https://www.python.org/ftp/python/3.12.10/) |
| pip 25.2 | [MIT](https://github.com/pypa/pip/blob/25.2/LICENSE.txt), bundled dependency notices remain upstream |
| PyTorch 2.2.1 | [BSD-style project license](https://github.com/pytorch/pytorch/blob/v2.2.1/LICENSE); bundled vendor components have additional terms |
| torchvision / torchaudio | [torchvision source](https://github.com/pytorch/vision/tree/v0.17.1), [torchaudio source](https://github.com/pytorch/audio/tree/v2.2.1), their respective BSD-style licenses |
| CUDA/NVRTC/cuDNN libraries in upstream wheels | [NVIDIA CUDA 11.8 EULA](https://docs.nvidia.com/cuda/archive/11.8.0/eula/index.html); applicable NVIDIA component terms are not replaced here |
| ComfyUI-Zluda | [GPL-3.0 source snapshot](https://github.com/patientx/ComfyUI-Zluda/tree/c0c011dca91b595091a4b032770da366fc90fb2b); preserve its GPL notices |
| ZLUDA fork | [MIT / Apache-2.0 upstream licenses](https://github.com/lshqqytiger/ZLUDA/tree/5e717459179dc272b7d7d23391f0fad66c7459cf); downloaded from the pinned upstream release |
| Patched gfx803 rocBLAS | [Patch publisher / source release](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/releases/tag/v1.0.0); [rocBLAS license](https://github.com/ROCm/rocBLAS/blob/rocm-5.7.1/LICENSE.md). No third-party patch binary is mirrored here |
| HIP SDK 5.7.1 | [AMD SDK terms](https://www.amd.com/en/developer/resources/rocm-hub/eula/licenses.html), [official SDK page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) |
| 7-Zip extraction utility | [License, including LGPL and unRAR restrictions where applicable](https://www.7-zip.org/license.txt), [source](https://www.7-zip.org/download.html); downloaded only when needed |
| ONNX Runtime DirectML | [MIT](https://github.com/microsoft/onnxruntime/blob/v1.24.4/LICENSE), upstream package/component notices |
| OpenVINO | [Apache-2.0](https://github.com/openvinotoolkit/openvino/blob/master/LICENSE), downloaded package notices |
| Microsoft VC++ runtime prerequisite | [Official publisher download](https://aka.ms/vs/17/release/vc_redist.x64.exe); automatically downloaded and installed if missing, under Microsoft's terms; not redistributed here |

## Important vendor boundaries

AMD's [deployment guidance](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/deployment-guidelines.html)
states that applications use the HIP runtime supplied by the installed GPU
driver. Accordingly, this project never mirrors or installs `amdhip64.dll`,
`atikmdag.sys`, or any other local driver/system file. SDK-only administrative
extraction is used; AMD Setup and driver packages are not executed.

NVIDIA CUDA 11.8 EULA section 1.2(8) restricts translating SDK-generated output
artifacts to non-NVIDIA targets. Distribution of an open-source installer does
not settle every recipient's rights for third-party tools or translated
artifacts. The experimental AMD/ZLUDA recipe is not a declaration that those
restrictions disappear; recipients must determine and obtain the rights they
need. This project does not mirror NVIDIA binaries or represent vendor approval.

The clean release is deliberately an online source installer rather than a
repacked vendor-runtime bundle. It also does not distribute the original
personal offline archive, which included a model and private diagnostic material.

## Privacy boundary

Only sanitized numeric measurements are published. The original prompts, logs,
host paths, serial/device identifiers, user profiles, credentials, and caches
remain outside this repository and its release. The installer has no sign-in,
stored credential, telemetry-upload or API-key configuration feature. Normal
publisher downloads and local application interfaces are distinct from an API key.

## Optional source contribution and PyTorch DirectML

The small-GEMM source contribution and portable adaptations are GPL-3.0-only;
see patches/small-gemm/NOTICE.txt. It is inspired by the linked Schaka finding,
not a redistribution of a Linux ROCm binary or a claim of a complete port.
Builds keep the separately acquired original rocBLAS binary unchanged; neither
that backend nor generated shim/hsaco/import-library artifacts are published here.

Optional torch-directml is downloaded from its publisher's pinned PyPI wheel.
See [Microsoft DirectML](https://github.com/microsoft/DirectML/tree/master/PyTorch)
and the upstream package's license/component notices. Python 3.11.9, Torch CPU
2.4.1 and torchvision CPU 0.19.1 retain their upstream notices. The source-only
project archive does not mirror those wheels, AMD runtime DLLs or model weights.
