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
