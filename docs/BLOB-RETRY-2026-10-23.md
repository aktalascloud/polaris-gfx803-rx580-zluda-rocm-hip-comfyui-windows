# gfx803 rocBLAS retry: 2026-10-23

The user explicitly authorized fresh tests for this retry. Tests ran in isolated
folders/processes; the default installation, baseline kernels, GPU driver,
registry and existing services were preserved. [Sanitized measurements](../data/blob-retry-2026-10-23.json).

## Correctness first

- Matched advanced-lvl-up baseline: eight FP32/FP16 matrix checks passed in both processes.
- Full likelovewant **optic** pair: one complete eight-check pass; reverse-order repeat crashed with native `0xc0000005` before the first FP16 result, after four FP32 results. A successful first run is not a stable compatibility claim.
- Full **vega10-fallback** pair: native `0xc0000005` before the first numerical result.
- Baseline DLL + optic kernels, baseline DLL + vega kernels: native `0xc0000005` before the first result.
- Optic DLL + baseline kernels, vega DLL + baseline kernels: first FP32 check failed; maximum absolute error **2.829238176345825**. These mixed sets are rejected.
- Optic with application BLAS batch 512 also crashed before its first control request. Batching is not enabled for the ZLUDA fixture/default by this retry.

A DLL and its matching kernel directory must be treated as one set. The installer
continues to hash-check and install the complete advanced-lvl-up archive. No
rejected candidate or arbitrary subset is copied into the working installation.

## Application comparison

| Configuration | GPU layers | Batch setting | Median decode, tokens/s | Median request, s | Controls |
|---|---:|---:|---:|---:|---|
| baseline-no-batch | 24 | -1 | 8.25 | 20.19 | 3/3 |
| optic-no-batch | 24 | -1 | 8.10 | 20.42 | 3/3 |
| optic-batch512 | 24 | 512 | N/A | N/A | Failed before controls |
| vulkan-no-batch | 24 | -1 | 9.58 | 17.42 | 3/3 |
| vulkan-batch512 | 24 | 512 | 9.25 | 14.71 | 3/3 |
| baseline-32layers | 32 | -1 | 8.33 | 19.96 | 3/3 |
| vulkan-32layers | 32 | 512 | 11.99 | 11.44 | 3/3 |
| optimized-vulkan-current | 37 | 1024 | 33.09 | 3.93 | 3/3 |

The first five rows use the same **1.95.1 oldpc archive**, same Qwen3-4B IQ4_XS
model, 24/37 GPU layers, eight CPU/BLAS threads, context 1024, greedy sampling
and three requests capped at 128 output tokens per successful configuration.
ZLUDA uses `normal 0 nommq`, disabled pinned host allocation and BLAS batch -1.
The bundled Vulkan native DLL is `koboldcpp_vulkan_noavx2.dll`; different native
backend/compiler paths are part of this practical comparison. Both successful
ZLUDA variants answered the three simple controls correctly.

32-layer rows use two requests each with eight threads; they explore offload
count rather than changing blobs. The `optimized-vulkan-current` row uses the
separate author's [optimized Vulkan build](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa),
version 1.123.1, 37/37 GPU layers, six CPU / four BLAS threads, batch/ubatch 1024
and its existing custom Vulkan runtime policy. Its ratio against ZLUDA is a
**whole-configuration result**, not an isolated ROCm/rocBLAS patch improvement.
The toolkit does not bundle or install that benchmark application.

Each displayed median summarizes the recorded requests, not independent random
sessions. Prefix caching was not disabled; reported prompt rates use the
application's processed-token accounting. Background desktop services stayed
running. Token rate is decode-only; request time includes prefill and API overhead.
The optimized build reports about 6500 prompt tokens/s for a very short cached-prefix request; the tiny timing interval is not a sustained-prefill benchmark. No such prefill claim is advertised. This is a limited single-card experiment, not a universal speed guarantee.

## Diffusion-like FP32 GEMMs

| [m,n,k], FP32 | Baseline median, ms | Optic median, ms | Relative GEMM throughput |
|---|---:|---:|---:|
| [320, 4096, 320] | 0.3715 | 0.8110 | -54.19% |
| [1280, 1024, 1280] | 1.0400 | 3.0684 | -66.11% |
| [320, 4096, 2880] | 2.0703 | 6.1726 | -66.46% |
| [768, 77, 768] | 0.2839 | 0.2840 | -0.05% |

Two processes/profile in baseline/optic/optic/baseline order; each row per process
uses five warmups and seven samples of five GEMMs, with final GPU synchronization.
CPU reference and transfers are excluded from timing. These are dense matrix
shapes inspired by diffusion workloads, **not actual UNet convolutions, a complete
ComfyUI workflow, images/s or proof of twofold image-generation acceleration**.
Numerical status and all sample timings are in the JSON. Optic's FP16 instability
remains relevant even if a selected FP32 shape runs successfully.

## Reproduction and installed patches

`tools/benchmark_blobs.py` is an explicit manual probe, never run by Install.cmd.
Use the installed CUDA Python environment and isolated component folders:

```powershell
runtime\envs\cuda\python.exe -I tools\benchmark_blobs.py --hip runtime\hip --zluda runtime\zluda --dll runtime\hip\bin\rocblas.dll --kernels runtime\hip\bin\rocblas\library --label baseline --cache C:\AI\Blob-Probe-Cache
# Add --suite diffusion for the FP32 diffusion-like matrix shapes.
```

For application reproduction, use the exact release/model/native DLL hashes,
settings and per-request counts in the JSON, port-bound only to 127.0.0.1. The
synthetic generation request asks for a detailed vegetable-garden guide;
three trials begin with A/B/C respectively to change the user-prefix token.
No personal prompts, raw worker logs, private paths, binaries, model weights,
credentials or compiled caches are published.

The installed patches remain: matched gfx803 rocBLAS + blobs, pinned ZLUDA 3.9.5,
Torch CUDA-library aliases with original NVRTC preserved, local HIP/rocBLAS DLL
loading, conservative Torch fallbacks and optional ComfyUI startup source patch.
[Full catalogue](PATCHES.md). Linux Schaka patches are not Windows DLL patches.
There is no newly validated fast-blob patch to promote from this retry.

The general installer has **not** undergone a fresh end-to-end installation.
ComfyUI still has **no image-generation validation**. The driver was not changed.
