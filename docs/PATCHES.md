# Automatically applied patches and compatibility settings

| Component | Installed change | Isolated speed/token gain |
|---|---|---|
| HIP SDK 5.7.1 | Original SDK extracted without running AMD Setup; existing GPU driver preserved | Not measured |
| advanced-lvl-up rocBLAS v1.0.0 | Replace rocblas.dll and the complete matching gfx803 kernel directory | Not measured; historical FP32/FP16 checks passed |
| ZLUDA 3.9.5 | Pinned upstream ROCm5 build; not a locally rebuilt/patched ZLUDA | Not measured |
| PyTorch 2.2.1+cu118 | ZLUDA aliases for cublas64_11.dll, cusparse64_11.dll and nvrtc64_112_0.dll; original NVRTC kept as nvrtc_cuda.dll | Compatibility change; not measured |
| Local loading | HIP_PATH, ROCBLAS_TENSILE_LIBPATH, local DLL search, DISABLE_ADDMM_CUDA_LT=1; Torch cuDNN/TF32 disabled | Not measured |
| Optional ComfyUI | [GPL source diff](../patches/comfy-offline-startup.diff) disables startup auto-pip and auto-update; conservative FP32/low-VRAM/CPU-VAE profile | Not measured; generation unverified |
| DirectML / OpenVINO | Separate NumPy 2.2.6 environment, distinct from Torch NumPy 1.26.4 | No paired timing; Radeon DirectML and OpenVINO CPU previously passed |

Validated rocblas.dll SHA256:
`9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d`.
Archive hashes and download URLs: [components.json](../profiles/components.json).
The DLL and its kernel directory must be used together.

## Historical numerical evidence

RX580 2048SP/gfx803; recorded 2026-10-22. Sanitized summary only, no raw logs.

| Check | Maximum absolute error | Outcome |
|---|---:|---|
| FP32 GEMM regression | 1.0609626770019531e-05 | Passed under its original tolerance |
| FP16 GEMM regression | 0.0100555419921875 | Passed under its original FP16 tolerance |
| Two-layer convolution network | 2.980232238769531e-07 | Passed |
| 60 BMM/copy iterations | 1.1444091796875e-05 | Passed |
| LayerNorm/Softmax, width 1024 | 9.313225746154785e-09 | Passed |

No new GPU test was performed to create this table. These are numerical checks,
not throughput measurements or certification for arbitrary shapes/applications.

## Excluded candidates and research

- likelovewant v0.5.7 gfx803-optic, gfx803-vega-fallback and common HIP5.7 candidates failed previous numerical checks; they are not installed.
- Tiny batched GEMM still has unsupported shapes/no Tensile solution.
- OpenVINO Ellesmere GPU returned incorrect zeros; use OpenVINO CPU.
- [Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803) Linux Wave64/small-GEMM/TC-coherence patches were studied, not applied to Windows HIP DLLs. Linux speedups must not be attributed to this package.
- No complete ComfyUI generation or isolated per-blob speed measurement is available.

The installed toolkit already applies the required Windows overrides. The repository
ships source patches and pinned acquisition logic, not a mirror of third-party binaries.
