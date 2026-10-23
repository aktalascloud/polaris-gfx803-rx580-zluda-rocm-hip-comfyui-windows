# Patch and workaround catalogue

The measured target is AMD Polaris gfx803. No isolated before/after performance
experiment exists for any individual change below. `N/A` means **not measured**,
not zero improvement. The percentages in the README compare whole CPU/GPU requests.

| Change | Purpose / observed outcome | Per-patch speed or token gain |
|---|---|---|
| advanced-lvl-up rocBLAS v1.0.0 DLL + matching gfx803 kernels | Restore usable GEMM kernels for Polaris. Validated DLL SHA256 `9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d`. FP32/FP16 checks passed earlier. | N/A; no valid stock-gfx803 throughput baseline |
| ZLUDA 3.9.5 / Torch 2.2.1+cu118 DLL aliases | Replace Torch `cublas64_11.dll`, `cusparse64_11.dll`, `nvrtc64_112_0.dll` with ZLUDA counterparts; preserve original NVRTC as `nvrtc_cuda.dll`. | N/A; compatibility change |
| Explicit local HIP/rocBLAS loading | Prevent loading an unrelated rocBLAS or silently selecting the wrong kernel folder. No global PATH or driver replacement. | N/A |
| `DISABLE_ADDMM_CUDA_LT=1`; AMD cuDNN/TF32/flash/memory-efficient SDP disabled | Avoid unsupported vendor-specific execution paths. Reuse the upstream ComfyUI-Zluda fallbacks, including CPU audio/TopK operations where that fork applies them. | N/A; no isolated timing |
| KoboldCPP `--blasbatchsize -1`, `nommq`, pinned-memory disabled | Normal batching gave unrelated text; MMQ failed. Disable BLAS batching while retaining real GPU offload. Three correct simple controls and four matching CPU/GPU continuations resulted. | N/A versus broken batching; measured complete-request CPU/GPU deltas are +1.94% to +54.14%, not a causal patch claim |
| ComfyUI offline-startup patch | Replace auto-pip with a package-version error and replace the auto-`git pull` launcher. Keep dependencies stable. [GPL diff](../patches/comfy-offline-startup.diff). | N/A; ComfyUI not launched |
| Conservative ComfyUI launch profile | FP32, lowvram, CPU VAE, sub-quadratic attention, no xformers/CUDA malloc async/external API nodes. | N/A; no images/s measurement |
| Separate ONNX environment | Keep NumPy 2.2.6 DirectML/OpenVINO separate from NumPy 1.26.4 Torch. DirectML physical Radeon passed; OpenVINO CPU passed. | N/A; no paired backend speed result retained |

## Negative results retained as findings, not bundled experiments

- ZLUDA Wave64 slow mode: RMS_NORM `invalid configuration argument`.
- Kobold MMQ path: `invalid device function`.
- KoboldCPP 1.123.1: GPU allocation `operation not supported` for a 2159.88 MiB buffer.
- likelovewant v0.5.7 gfx803-optic, gfx803-vega-fallback and common HIP5.7 libraries:
  numerical checks failed. The downloaded setup uses the validated advanced-lvl-up library.
- `GPU_FORCE_BLIT_COPY_SIZE=100000`: did not fix the incorrect Kobold text on this Windows stack.
- Tiny batched GEMM `[3,1,2] × [3,2,1]`: no Tensile solution.
- OpenVINO GPU/Ellesmere: incorrect zero MatMul; CPU is the documented target.
- Full 250-convolution Torch inference was not confirmed; its initial compilation was stopped.

## Research references, not installed Windows patches

[Schaka/rocm-gfx803](https://github.com/Schaka/rocm-gfx803) contains Linux ROCm
Wave64, small-GEMM and TC-coherence fixes. Those are not Windows HIP 5.7 DLL
replacements and were not applied to CUDA PyTorch here. Their reported Linux
speedups are **not measurements from this project**. The Windows reuse probe
passed 60 BMM/copy iterations; LayerNorm/Softmax widths 1024/2560/9728 also passed.

## Pascal

Pascal uses native CUDA, not ZLUDA or AMD rocBLAS. No Pascal card was measured in
this session. Do not infer Pascal token rates, percentage gains, successful
ComfyUI generation, or universal driver support from the RX580 findings.
