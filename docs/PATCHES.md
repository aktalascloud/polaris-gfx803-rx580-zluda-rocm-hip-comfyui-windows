# Patch catalogue

| Component | Change | Evidence / limit |
|---|---|---|
| HIP SDK 5.7.1 | SDK-only extraction; original driver retained | Fresh installation tested with an existing SDK |
| advanced-lvl-up rocBLAS | Matched DLL and gfx803 kernel directory | Stable default; do not mix with optic/vega |
| ZLUDA 3.9.5 | Pinned upstream ROCm5 build | Not a locally rewritten ZLUDA |
| Torch CUDA aliases | cublas/cusparse/NVRTC replacements; original NVRTC preserved | Hash/alias checks passed |
| Local runtime setup | Process-local HIP/PATH, explicit existing x64 HIP runtime, cuDNN/TF32 fallbacks | Current System32 runtime tested; other versions unverified |
| Small-GEMM, opt-in | Tiny FP32 HIP kernel; four APIs overridden, 3932 exports forwarded; LLD ordinal repair | Baseline strided: 63/69; patched: 69/69 on four APIs; all 3936 original ordinals retained |
| Torch DirectML, opt-in | Separate cp311 / Torch 2.4.1+cpu environment | FP32/FP16 matmul and conv2d passed |
| ONNX / OpenVINO | Separate NumPy 2 environment, physical Radeon selection, CPU-only OpenVINO | Strict profiled DirectML and OpenVINO CPU passed |
| ComfyUI | [Startup diff](../patches/comfy-offline-startup.diff), pinned dependencies, FP32/low-VRAM/CPU-VAE preset | Installed; startup/generation unverified |
| Automatic attention | Stock small FP32; bounded large/masked/FP16 inference, masks/upcast/layouts preserved | [146 GPU cases and tensor-level timing/memory evidence](ATTENTION.md); no generation claim |
| Diagnostics | Model-free CUDA/PTX check; separate DXGI/Torch device enumeration | 64/64 vector elements; software adapters excluded |

Default backend SHA256: `9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d`.
Exact download URLs/hashes: [components.json](../profiles/components.json).

## Results and attribution

- [Small-GEMM](SMALL-GEMM.md): build, guards, ABI, rollback and current validation.
- [Blob retry](BLOB-RETRY-2026-10-23.md): optic instability, rejected vega/mixed sets and measured comparisons.
- [Current evidence](../data/small-gemm-validation-20261023.json) and [source-machine history](../data/small-gemm-source-machine.json) remain separate.

The unmodified backend still has unsupported tiny shapes. The optional Windows
small-GEMM workaround addresses its tested subset; it is inspired by
[Schaka's finding](https://github.com/Schaka/rocm-gfx803), not a direct application
of Linux ROCm patches. OpenVINO Ellesmere GPU returned incorrect zeros; use CPU.
No isolated generation-speed gain is established by these correctness checks.

## Upstream review — 2026-11-17

- **ComfyUI:** New installs ignore empty or comment-only extra model path YAML using the fix proposed in [ComfyUI #17768](https://github.com/Comfy-Org/ComfyUI/pull/17768). The Windows DLL search-path fix in [ComfyUI-Zluda #401](https://github.com/patientx/ComfyUI-Zluda/pull/401) is already covered by the toolkit's launcher. The [fork snapshot reviewed](https://github.com/patientx/ComfyUI-Zluda/tree/8eb6fd2701758b2554e5785bbac52ba4ceee2a45) documents PyTorch 2.7 as the minimum, above the pinned 2.2.1.
- **ZLUDA:** [lshqqytiger #132](https://github.com/lshqqytiger/ZLUDA/pull/132) adds v3.9.6 with CUDA 12.4 and still publishes a ROCm 5 Windows build. Its compatibility with the pinned CUDA 11.8 Torch and gfx803 rocBLAS is unverified. [ZLUDA v7-preview.21](https://github.com/vosen/ZLUDA/releases/tag/v7-preview.21) adds a Windows launcher picker. The open [FP16 conversion PR](https://github.com/vosen/ZLUDA/pull/685) reports gfx1100 validation.
- **ROCm:** The matched [advanced-lvl-up bundle](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/releases/tag/v1.0.0) is still its newest release. The [HIP 7.1.1 community bundle](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases/tag/v0.7.1.1) lists no gfx803 support. [Schaka #8](https://github.com/Schaka/rocm-gfx803/pull/8) proposes a native Linux ROCm/PyTorch TopK fallback, outside this Windows CUDA build.

The pinned source installer and attention apply/rollback/reapply passed. The YAML check covered empty, comment-only, explicit null and a valid relative model path (4/4). This review ran no GPU workloads or full ComfyUI startup.
