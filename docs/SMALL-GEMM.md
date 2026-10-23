# Optional small FP32 GEMM fix and backend diagnostics

This is a **Windows/HIP source workaround inspired by Schaka's small-GEMM finding**,
not a port of the Linux ROCm patch set. It restores six missing tiny FP32 cases on
the tested RX580. It does not establish an increase in tokens/s or images/s.
The original matched advanced-lvl-up backend remains the default.

## Install into a fresh private runtime

From this source checkout, after reading THIRD_PARTY_NOTICES.md:

```powershell
.\Install.cmd -InstallRoot C:\AI\Polaris-New -WithComfyUI -WithTorchDirectML -WithSmallGemmPatch -AcceptThirdPartyLicenses
# Optional existing SDK source: -HipRoot "C:\Program Files\AMD\ROCm\5.7"
# Optional adapter choice: -DxgiDevice <DXGI-index>
# Optional existing runtime: -HipRuntime "C:\Existing-HIP\amdhip64.dll"
```

Both `-WithSmallGemmPatch` and `-WithTorchDirectML` are **off by default**.
ComfyUI retains its existing explicit switches/default prompt. Existing destinations
are refused; do not point setup at a working ZLUDA/ComfyUI installation.
The installer builds the fix from source in an isolated staging directory and
verifies the original backend hash, generated files, export names and ordinals.
It installs no model, runs no numerical GPU workload and changes no driver,
registry, TDR, firmware or global PATH. `-DownloadCache` reuses only verified
complete component files; `.part` files never count as ready downloads.

The supported small-GEMM path must contain **ASCII characters**; spaces work and
were tested. The native loader currently uses ANSI Win32 paths. The generic
runtime features do not inherit a claim of native Unicode-path support.
The SDK must provide clang, clang-offload-bundler and lld-link. Build output must
be new and outside the source SDK. The installer uses `work/` outside its runtime.

## Scope and installed files

Only these four APIs are overridden:

- `rocblas_sgemm`
- `rocblas_sgemm_strided_batched`
- `rocblas_gemm_ex`
- `rocblas_gemm_strided_batched_ex`

The intercepted case is FP32, **1 <= M,N,K <= 8**, host alpha/beta, recognized
Polaris RX470/480/570/580/590 at HIP device **0**, and the initial HIP context.
Ex requires FP32 A/B/C/D/compute and default algorithm/solution/flags.
Transpose/conjugate-transpose on real inputs, leading dimensions, broadcast input
strides, separate Ex C/D and the handle's stream are respected.

FP16, larger matrices, device scalars, zero/invalid sizes, memory-size queries,
graph capture, other devices and special algorithms are delegated to the original
backend. Device/context guards fail closed. A different HIP context does not reuse
the cached kernel; complete context destruction/recreation and concurrent callers
remain **unvalidated**. Do not generalize this single-context workaround to all
GPU/context combinations. For capture, only status forwarding/capture creation
was checked; no captured graph was launched in that contract test.

The private HIP bin directory contains:

| File | Purpose |
|---|---|
| `rocblas.dll` | Built forwarding shim |
| `rocblas_backend.dll` | Unmodified matched original backend |
| `rocblas_small_gemm.hsaco` | HIP gfx803 code object |
| `small-gemm-patch.json` | Installed hashes and enabled/disabled receipt |
| `rocblas/library/` | Unchanged matched kernel directory |

Backend SHA256 is fixed:
`9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d`.
Never mix it with optic/vega kernels. The generator refuses a shim as backend.
It forwards **3932 original named exports**, overrides four, and adds one version
export. All **3936 original ordinals** are preserved. AMD LLD 17 was observed to
reassign forwarder ordinals; the build repairs the generated PE export table in
place, then verifies it independently. The loader check resolves all 3937 names
and ordinals. Timestamp-dependent shim hashes can differ between builds.

`build/imports/kernel32.dll` is an import-library construction artifact. It is
**never installed or distributed** as a replacement Windows DLL. Only the three
runtime binaries listed above plus the receipt are installed locally. Git and
source packages contain their source, not any compiled DLL/code object.

## Rebuild, apply and roll back

Close every consumer of the target private runtime first. DLLs held by another
process are refused; no forced overwrite or automatic rollback is attempted.

```powershell
# If already patched, supply rocblas_backend.dll as the trusted original.
powershell -NoProfile -ExecutionPolicy Bypass -File patches\small-gemm\Build-SmallGemm.ps1 -HipRoot C:\AI\Polaris-New\hip -BackendDll C:\AI\Polaris-New\hip\bin\rocblas_backend.dll -Python C:\AI\Polaris-New\envs\diagnostics\python.exe -OutDir C:\AI\Polaris-Build-1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\Small-Gemm.ps1 -RuntimeRoot C:\AI\Polaris-New -BuildDirectory C:\AI\Polaris-Build-1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\Small-Gemm.ps1 -RuntimeRoot C:\AI\Polaris-New -Action Rollback
```

Rollback atomically restores the original backend as `rocblas.dll`, marks the
receipt disabled and keeps the matched kernel directory unchanged. The trusted
backup/code object are retained for inspection/reapplication; neither is used as
a patch after rollback. Repeating installation of an identical build or rollback
is idempotent. Windows PowerShell 5.1 uses an explicit null-string value for the
atomic replacement's optional backup argument; an ordinary `$null` was found to
be converted into an invalid empty path. That installer failure was fixed and
then the full fresh installation was repeated successfully.

## Optional PyTorch DirectML

`-WithTorchDirectML` adds `envs/torch-directml` and `Python-DirectML.cmd`:
Python **3.11.9**, Torch **2.4.1+cpu**, torchvision **0.19.1+cpu**,
torch-directml **0.2.5.dev240914**, NumPy **1.26.4**.
The `+cpu` Torch wheel supplies the compatible Torch core; `torch_directml`
provides the physical GPU device. This is separate from CUDA Torch 2.2.1+cu118
and the ONNX/OpenVINO NumPy 2 environment. All 13 selected DirectML-environment
wheel versions/hashes are pinned; primary binary URLs/ABI are in components.json.

```powershell
C:\AI\Polaris-New\Python-DirectML.cmd -c "print(dml_device); print((torch.ones(2,device=dml_device)+1).cpu())"
```

The launcher provides `torch`, `torch_directml`, and `dml_device` to `-c` and
script/module entrypoints. User models/tensors must explicitly use that device;
CUDA code is not automatically rewritten into DirectML code.

DXGI enumeration excludes software adapters and requires AMD vendor 0x1002.
When several physical AMD devices exist, list/choose one using `-DxgiDevice` or
`AI_DXGI_DEVICE`. **Torch DirectML indices are enumerated separately** and must
not be copied from DXGI. Matching uses hardware names, stripping a trailing NUL
returned by torch-directml on this machine. Ambiguous matching requires an
explicit `AI_TORCH_DML_DEVICE`. Actual multiple-GPU topology was unavailable;
index-namespace/ambiguity handling also has synthetic unit tests.

## Explicit checks and runtime selection

```powershell
C:\AI\Polaris-New\Check-Zluda.cmd
C:\AI\Polaris-New\envs\diagnostics\python.exe -I tools\diagnose.py --root C:\AI\Polaris-New
C:\AI\Polaris-New\envs\cuda\python.exe -I tests\check-gemm.py --hip-root C:\AI\Polaris-New\hip --dll C:\AI\Polaris-New\hip\bin\rocblas.dll --report work\gemm.json
# Other APIs: --api sgemm / ex / strided-ex; add --nondefault-stream.
# Optional --hip-runtime points at an existing local amdhip64.dll.
```

Check-Zluda uses an isolated minimal embedded Python with no Torch, OpenVINO or
ComfyUI dependency. It loads the selected ZLUDA nvcuda.dll, allocates/copies GPU
memory, compiles PTX and checks vector_add on 64 elements. Reports default to the
private runtime's temp directory, not the source checkout.

HIP runtime selection is explicit argument/environment (`AI_HIP_RUNTIME`), then
profile, then existing System32 runtime, then existing SDK-local runtime. Explicit
missing/wrong-architecture paths fail rather than silently switching runtimes.
Paths, x64 PE architecture, file version and SHA256 are inspectable. No arbitrary
HIP/driver DLL is downloaded or redistributed. These controls are used by
Python-CUDA and Check-Zluda; a foreign native program launched by generic
Run-ZLUDA still owns its dependency loading and is **not certified** to honor a
Python runtime override. HIP-Shell provides process-local HIP/PATH only.

On the current machine, automatic System32 selection and explicit selection of
the **same** installed DLL were tested. A second runtime/old driver was not
available. The source-machine Windows Server/private-runtime result is separately
recorded in data/small-gemm-source-machine.json and is not a fresh validation of
that combination here. OpenVINO remains **CPU only** in this toolkit.

## Validation and limits

See [current machine evidence](../data/small-gemm-validation-20261023.json).
Current platform: Windows 11 Pro build 26200, RX580X / RX580 2048SP / gfx803,
AMD driver 31.0.14037.18001, HIP SDK 5.7.1, existing amdhip64.dll file version
10.0.3592.0. Original working ZLUDA/ComfyUI and protected driver files were preserved.

The current checks include build/export ABI, native GEMM baseline and patched
runs, scalar/padding/broadcast/stream/mode contracts, CUDA PTX/vector add, Torch
DirectML FP32/FP16 matmul+conv2d, strict profiled ONNX DirectML, OpenVINO CPU,
quoted launchers/exit codes, lock refusal, idempotency, rollback/reapply and a
fresh all-options installation using the existing SDK and verified download cache.
The native default strided suite has 69 cases / 98 comparison iterations; that
number is not a universal API-call count, since non-batched APIs make one call
per batch. Tolerances remain rtol=1e-4 and atol=3e-5 against NumPy FP64 reference.

Untested: downloading/extracting the whole AMD SDK instead of using -HipRoot;
installing VC++ when absent; desktop shortcut creation (-NoDesktopShortcut was
used); ComfyUI server startup/image generation; other HIP versions/drivers/cards;
real multi-GPU mapping; concurrent native calls and full context reset lifecycle;
native Unicode paths. No model download or text/image generation was performed.
No speedup in tokens/s/images/s is claimed from these correctness checks.

Source contribution license: GPL-3.0-only. Original contribution hashes and
historical results remain separate from modified portable source and current
measurements. See patches/small-gemm/NOTICE.txt and THIRD_PARTY_NOTICES.md.
