# Polaris patches and maintenance

## Default Windows profile

Install.cmd invokes tools/Install.ps1. It stages dependencies from profiles/, copies launchers/ to the runtime root and applies the patches below. With ComfyUI selected, it adds a desktop shortcut unless -NoDesktopShortcut is set. The runtime uses local HIP/DLL paths, preserves original NVRTC, disables external ComfyUI API nodes and uses low VRAM, FP32 and CPU VAE. The pinned snapshot lacks Z-Image support.

| Component | Change |
|---|---|
| HIP 5.7.1 / ZLUDA 3.9.5 | SDK-only extraction and pinned upstream ROCm5 build |
| rocBLAS | Matched advanced-lvl-up DLL and gfx803 kernels. SHA-256 of the DLL is 9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d |
| PyTorch | ZLUDA cublas, cusparse and NVRTC aliases, with original NVRTC preserved |
| ComfyUI | [Offline startup and empty YAML fix](comfy-offline-startup.diff). [Pinned download hashes](../profiles/components.json) |
| Attention | [Hash-guarded installer](attention_patch.py) and [tiled implementation](polaris_attention.py), enabled automatically for optional ComfyUI |
| Small-GEMM | [Windows HIP shim and kernel](small-gemm/), enabled by -WithSmallGemmPatch |

Pass -WithoutComfyUI to omit ComfyUI. For a local SDK use -HipRoot. -DownloadCache reuses hash-verified complete files. For Radeon selection pass -DxgiDevice or set AI_DXGI_DEVICE when ambiguous. Torch DirectML uses its own indices via AI_TORCH_DML_DEVICE and a separate cp311 / Torch 2.4.1+cpu environment. Use dml_device explicitly. OpenVINO uses CPU.

Python-CUDA and Check-Zluda select an explicit HIP runtime or AI_HIP_RUNTIME, then the saved profile, then existing System32 and SDK candidates. Invalid explicit Windows x64 DLL paths fail. The generic Run-ZLUDA launcher delegates dependency loading to the wrapped application. Use an ASCII runtime path for native Small-GEMM. Spaces were tested.

~~~powershell
$runtime = 'C:\AI\Polaris-New'
& "$runtime\envs\diagnostics\python.exe" -B -I "$runtime\tools\diagnose.py"
& "$runtime\Check-Zluda.cmd"  # manual GPU check
~~~

## Attention

Small ordinary FP32 uses ComfyUI stock math. Large (estimated score ≥8 MiB), masked and FP16 inference uses head/query tiles, with score tiles capped at 32 MiB and adjusted to available memory. FP16 computes in FP32 and returns the original dtype. Explicit backends, training, unsupported devices/dtypes, explicit no-upcast-half and keys too long for one score row retain upstream paths.

Boolean True permits attention. Additive masks accept -inf; fully masked rows with finite inputs are zero. Mask layouts are [K], [Q,K], [B,Q,K] and broadcastable [B,H,Q,K]. Batched padding uses [B,1,1,K]. Inputs and outputs support flattened or explicit head layout, including unequal value depth.

~~~powershell
& "$runtime\envs\diagnostics\python.exe" -B -I "$runtime\patches\attention_patch.py" status "$runtime\apps\ComfyUI"
# Close consumers before rollback. Use install to reapply.
& "$runtime\envs\diagnostics\python.exe" -B -I "$runtime\patches\attention_patch.py" rollback "$runtime\apps\ComfyUI"
~~~

Rollback restores the verified upstream prefix. Modified or unknown source revisions are refused. The source install block, idempotent apply, rollback and reapply were validated without starting the server.

## Small-GEMM

Tiny FP32 (M,N,K from 1 to 8), host alpha/beta, Polaris device 0 and initial HIP context. Four overrides cover sgemm and gemm_ex, including strided batched variants. Ex requires FP32 A/B/C/D/compute and default algorithm, solution and flags. The shim forwards requests outside those bounds to the original backend. All 3936 original export ordinals are preserved, including 3932 forwarded. The shim adds one version export.

Close consumers before apply/rollback. Build against the trusted original backend. Use rocblas_backend.dll if already patched or rocblas.dll from an unpatched matched runtime. The generated kernel32 import stub is a build-only artifact.

Small-GEMM builds the gfx803 kernel at `-O3` with `-fno-fast-math`. The x64 shim uses `-O3 -mtune=haswell` and full LTO at level 3. Haswell scheduling tunes for Xeon E5 v3 while keeping the baseline x64 instruction set. The HIP 5.7.1 build and all original export ordinals were verified with these flags. The new build has no GPU timings or numerical results.

~~~powershell
$build = 'C:\AI\Polaris-Build'
powershell -NoProfile -ExecutionPolicy Bypass -File patches\small-gemm\Build-SmallGemm.ps1 -HipRoot "$runtime\hip" -BackendDll "$runtime\hip\bin\rocblas_backend.dll" -Python "$runtime\envs\diagnostics\python.exe" -OutDir $build
powershell -NoProfile -ExecutionPolicy Bypass -File patches\small-gemm\Small-Gemm.ps1 -RuntimeRoot $runtime -BuildDirectory $build
powershell -NoProfile -ExecutionPolicy Bypass -File patches\small-gemm\Small-Gemm.ps1 -RuntimeRoot $runtime -Action Rollback
~~~

Rollback restores the original DLL hash and disables the receipt. The original DLL and code object support reapply. Keep the DLL and matched kernel directory together. Preserve the [native attribution](small-gemm/NOTICE.txt).

## Recorded validation

[Numeric evidence](../tests/results.json) and [reproduction scripts](../tests/) are grouped in `tests/`. The JSON separates attention, blob trials, and Small-GEMM results from the two machines. GPU checks and benchmarks require explicit authorization.

- Attention used an independent CPU FP64 reference. 53 CPU cases, 146 GPU cases per process (two processes), four guards and seven selector checks passed. The 4096×4096 result uses 18 alternating warm samples per algorithm. Smaller stock calls have minor dispatch overhead. Tensor allocation peaks are additional PyTorch allocations, excluding existing inputs, weights, reserved cache and driver memory.
- The October Small-GEMM build passed 69/69 per API (max error 3.34e-6), export checks, rollback and reapply. CUDA/PTX, DirectML and OpenVINO CPU checks also passed on the reference machine. Other runtimes, real multi-GPU, concurrent native calls, complete context reset and native Unicode paths remain unverified.
- The matched rocBLAS baseline passed both eight-case runs. Optic failed on repeat and at batch 512, vega failed before its first result, and mixed DLL/kernel pairs failed. On the controlled app fixture (24 layers, batch -1), baseline ZLUDA reached 8.25 tokens/s, optic 8.10 and Vulkan 9.58. The separately tuned Vulkan setup reached 33.09 tokens/s with different offload and batch settings. These are fixture-level comparisons. No full-model attention, generation, training or universal speed claim is supported.

## Upstream review — 2026-11-17

- **ComfyUI:** Installs ignore empty or comment-only extra model path YAML using the fix proposed in [ComfyUI #17768](https://github.com/Comfy-Org/ComfyUI/pull/17768). The Windows DLL search-path fix in [ComfyUI-Zluda #401](https://github.com/patientx/ComfyUI-Zluda/pull/401) is already covered by the toolkit's launcher. The [fork snapshot reviewed](https://github.com/patientx/ComfyUI-Zluda/tree/8eb6fd2701758b2554e5785bbac52ba4ceee2a45) documents PyTorch 2.7 as the minimum, above the pinned 2.2.1.
- **ZLUDA:** [lshqqytiger #132](https://github.com/lshqqytiger/ZLUDA/pull/132) adds v3.9.6 with CUDA 12.4 and still publishes a ROCm 5 Windows build. Its compatibility with the pinned CUDA 11.8 Torch and gfx803 rocBLAS is unverified. [ZLUDA v7-preview.21](https://github.com/vosen/ZLUDA/releases/tag/v7-preview.21) adds a Windows launcher picker. The open [FP16 conversion PR](https://github.com/vosen/ZLUDA/pull/685) reports gfx1100 validation.
- **ROCm:** The matched [advanced-lvl-up bundle](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/releases/tag/v1.0.0) is still its newest release. The [HIP 7.1.1 community bundle](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.7.1.1) lists no gfx803 support. [Schaka #8](https://github.com/Schaka/rocm-gfx803/pull/8) proposes a native Linux ROCm/PyTorch TopK fallback, outside this Windows CUDA build.

The pinned source installer and attention apply/rollback/reapply passed. The YAML check covered empty, comment-only, explicit null and a valid relative model path (4/4). This review ran no GPU workloads or full ComfyUI startup.

## Licenses

Packaging scripts and documentation © 2026 aktalascloud, [GPL-3.0-only](../LICENSE). The startup diff modifies the pinned ComfyUI fork and preserves upstream notices. The bounded attention rewrite was inspired by the [advanced-lvl-up GPL helper](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/blob/6547664db8732305530b969b54c9b7d3806c02e1/jab-attention-VRAM-helper.zip). The [Small-GEMM notice](small-gemm/NOTICE.txt) preserves original provenance. Setup configures no cloud keys and provides no vendor endorsement.

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

### Vendor limits

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
