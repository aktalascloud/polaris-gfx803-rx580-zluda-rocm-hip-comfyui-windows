# Small-GEMM and runtime guide

Optional Windows/HIP workaround for tiny FP32 GEMMs, inspired by Schaka's finding.
The matched original backend stays the default. No generation-speed gain is claimed.

## Install and use

Run from the source checkout into a **new** private runtime:

```powershell
.\Install.cmd -InstallRoot C:\AI\Polaris-New -WithComfyUI -WithTorchDirectML -WithSmallGemmPatch
# Optional: -HipRoot "C:\Program Files\AMD\ROCm\5.7" -DxgiDevice <index>
# Optional: -HipRuntime "C:\Existing-HIP\amdhip64.dll" -NoDesktopShortcut
```

`-WithSmallGemmPatch` and `-WithTorchDirectML` default to off. The installer downloads
pinned packages, builds locally and verifies backend/file hashes and export ordinals.
It never runs numerical GPU tests, downloads models or replaces a GPU driver.
`-DownloadCache` accepts only verified complete files, never unfinished `.part` files.

Use an ASCII path; **spaces are tested, native Unicode paths are not supported**.
Build staging must be new and outside the SDK/private runtime. SDK tools required:
clang, clang-offload-bundler and lld-link. The generated `imports/kernel32.dll`
exists only to build an import library: it is never installed as a Windows DLL.

## Native scope

Four overrides: `rocblas_sgemm`, `rocblas_sgemm_strided_batched`, `rocblas_gemm_ex`,
`rocblas_gemm_strided_batched_ex`. Interception requires **FP32, 1≤M,N,K≤8**, host
alpha/beta, recognized Polaris at HIP device **0**, and the initial HIP context.
Ex additionally requires FP32 A/B/C/D/compute and default algorithm/solution/flags.
Transposes, leading dimensions, broadcast input strides, separate Ex C/D and the
handle's stream are covered. Larger/FP16 cases, device scalars, query/capture modes,
other devices and special algorithms are forwarded to the original backend.

| Private runtime file | Purpose |
|---|---|
| `hip/bin/rocblas.dll` | Built forwarding shim |
| `hip/bin/rocblas_backend.dll` | Unmodified matched backend |
| `hip/bin/rocblas_small_gemm.hsaco` | gfx803 kernel |
| `hip/bin/small-gemm-patch.json` | Hashes and enabled/disabled receipt |
| `hip/bin/rocblas/library/` | Unchanged matched kernels |

Trusted backend SHA256:
`9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d`.
Never mix optic/vega kernels or pass a shim as the original backend. All **3936
original ordinals** are preserved: 3932 exports forwarded, four overridden, one
version export added. AMD LLD 17 forwarder ordinals are repaired and independently
verified; build timestamps can change the resulting shim hash.

## Rebuild and rollback

Close all consumers first. Loaded DLLs are refused; no forced overwrite or automatic
rollback occurs. If already patched, use `rocblas_backend.dll`; otherwise use the
original `rocblas.dll`. Example rebuild:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File patches\small-gemm\Build-SmallGemm.ps1 -HipRoot C:\AI\Polaris-New\hip -BackendDll C:\AI\Polaris-New\hip\bin\rocblas_backend.dll -Python C:\AI\Polaris-New\envs\diagnostics\python.exe -OutDir C:\AI\Polaris-Build
powershell -NoProfile -ExecutionPolicy Bypass -File tools\Small-Gemm.ps1 -RuntimeRoot C:\AI\Polaris-New -BuildDirectory C:\AI\Polaris-Build
powershell -NoProfile -ExecutionPolicy Bypass -File tools\Small-Gemm.ps1 -RuntimeRoot C:\AI\Polaris-New -Action Rollback
```

Rollback atomically restores the original hash and disables the receipt. The matched
kernel directory stays unchanged; the trusted backup/code object remain for reapply.
Repeated install/rollback is idempotent. The build's backend/code object is never
published in Git or source ZIPs. Source licensing: GPL-3.0-only; see [NOTICE](../patches/small-gemm/NOTICE.txt).

## DirectML, adapters and HIP runtime

Torch DirectML is isolated: **Python 3.11.9, Torch 2.4.1+cpu, torchvision 0.19.1+cpu,
torch-directml 0.2.5.dev240914, NumPy 1.26.4**. All 13 wheel hashes are pinned.
The CPU-core Torch wheel uses the Radeon through `torch_directml`; user tensors/models
must explicitly use `dml_device`. CUDA code is not rewritten automatically.

```powershell
C:\AI\Polaris-New\Python-DirectML.cmd -c "print(dml_device); print((torch.ones(2,device=dml_device)+1).cpu())"
C:\AI\Polaris-New\Check-Zluda.cmd
C:\AI\Polaris-New\envs\diagnostics\python.exe -I tools\diagnose.py --root C:\AI\Polaris-New
```

DXGI selection excludes software and requires AMD vendor `0x1002`. Use `-DxgiDevice`
or `AI_DXGI_DEVICE` when ambiguous. Torch DirectML enumerates **its own indices**;
do not copy DXGI indices. Ambiguous name matching needs `AI_TORCH_DML_DEVICE`.

Python-CUDA/Check-Zluda choose an explicit argument or `AI_HIP_RUNTIME`, then profile,
then existing System32 runtime, then existing SDK runtime. Explicit missing/wrong-x64
paths fail without fallback. The selected file's version/SHA256 is inspectable;
no arbitrary HIP/driver DLL is downloaded. Generic Run-ZLUDA launches foreign code
whose dependency loading is not certified to honor a Python runtime override.
HIP-Shell changes only its process environment.

## Reproduce and interpret checks

```powershell
C:\AI\Polaris-New\envs\cuda\python.exe -I tests\check-gemm.py --hip-root C:\AI\Polaris-New\hip --dll C:\AI\Polaris-New\hip\bin\rocblas.dll --report work\gemm.json
# Other APIs: --api sgemm / ex / strided-ex; add --nondefault-stream.
# Add --hip-runtime <existing-amdhip64.dll> when required.
```

[Current-machine evidence](../data/small-gemm-validation-20261023.json): Windows 11
Pro build 26200, RX580X/2048SP gfx803, driver 31.0.14037.18001, SDK 5.7.1,
existing runtime file version 10.0.3592.0. The baseline strided run has six unsupported cases;
the patch passes **69/69 per API**, max error `3.337860107421875e-6` with unchanged
`rtol=1e-4, atol=3e-5` against NumPy FP64. The 98 comparison iterations are not a
universal API-call count: non-batched APIs call once per batch.

Also checked: scalar/padding/broadcast/stream contracts; device scalars; invalid
arguments; query/capture status forwarding; Windows name/ordinal resolution;
CUDA PTX/vector_add 64/64; Torch DirectML FP32/FP16; strict ONNX DirectML; OpenVINO
CPU; quoted launchers/exit codes; lock refusal, rollback/reapply; fresh all-options
installation with an existing SDK and verified cache. Working stacks and five
protected driver files were unchanged. No models or generation were used.

**Unverified:** other runtimes/old drivers; real multi-GPU mapping; concurrent native
calls and complete context reset; native Unicode paths; SDK network extraction;
missing-VC++ installation; desktop shortcut creation; ComfyUI startup/generation.
Graph capture was created/status-checked, not executed. Source-machine Server 2022 /
private-runtime evidence is [separate](../data/small-gemm-source-machine.json).
OpenVINO stays CPU-only. See [third-party terms](../THIRD_PARTY_NOTICES.md).
