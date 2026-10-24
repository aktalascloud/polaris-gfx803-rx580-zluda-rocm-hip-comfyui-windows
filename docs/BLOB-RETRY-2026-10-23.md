# Blob comparison · 23 October 2026

Isolated RX580 2048SP/gfx803 tests; original runtime, driver and existing services
preserved. [Full numeric evidence](../data/blob-retry-2026-10-23.json).

## Decision

Keep the matched **advanced-lvl-up DLL + kernels**. No safe faster replacement was found.

| Candidate | Observed outcome |
|---|---|
| Baseline | Both eight-case FP32/FP16 runs passed |
| Full optic | First run passed; repeat crashed before FP16; application batch 512 also crashed |
| Full vega fallback | Crashed before first numeric result |
| Baseline DLL + optic/vega kernels | Both crashed |
| Optic/vega DLL + baseline kernels | Both failed FP32; max error `2.829238176345825` |

Mixed pairs are rejected. The later [optional small-GEMM source fix](SMALL-GEMM.md)
is a separate correctness change, not a successful optic/vega replacement.

## Application fixture

| Configuration | GPU layers | Batch | Decode, tokens/s | Request, s |
|---|---:|---:|---:|---:|
| Baseline ZLUDA | 24 | -1 | 8.25 | 20.19 |
| Optic ZLUDA | 24 | -1 | 8.10 | 20.42 |
| Optic ZLUDA | 24 | 512 | Crashed | — |
| Vulkan | 24 | -1 | 9.58 | 17.42 |
| Vulkan | 24 | 512 | 9.25 | 14.71 |
| Baseline ZLUDA | 32 | -1 | 8.33 | 19.96 |
| Vulkan | 32 | 512 | 11.99 | 11.44 |
| Optimized Vulkan | 37 | 1024 | 33.09 | 3.93 |

Values are per-configuration medians. The first five rows use KoboldCPP 1.95.1 oldpc,
the same Qwen3-4B IQ4_XS, eight CPU/BLAS threads, context 1024 and three greedy
requests capped at 128 output tokens. ZLUDA uses `normal 0 nommq` and no pinned
host allocation; bundled Vulkan uses its `noavx2` DLL. Successful variants passed
three simple controls. The application/model are measurement fixtures, not bundled components.

32-layer rows have two requests. The [optimized Vulkan build](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa)
uses version 1.123.1, 37/37 layers, six CPU/four BLAS threads, batch/ubatch 1024 and
custom kernels: its roughly 4.01× ratio is a **whole-configuration comparison**.
Optic's decode difference versus baseline is **-1.78%**.

## Diffusion-like FP32 GEMMs

| [M,N,K] | Baseline, ms | Optic, ms | Throughput change |
|---|---:|---:|---:|
| [320,4096,320] | 0.3715 | 0.8110 | -54.19% |
| [1280,1024,1280] | 1.0400 | 3.0684 | -66.11% |
| [320,4096,2880] | 2.0703 | 6.1726 | -66.46% |
| [768,77,768] | 0.2839 | 0.2840 | -0.05% |

Order: baseline/optic/optic/baseline; five warmups and seven samples of five GEMMs
per shape/process, with synchronization. CPU reference, transfers and initial
compilation are excluded. These are dense matrices, not ComfyUI workflows/images/s.

Samples are small, not fully randomized; desktop services and prefix caching
remained enabled. Short/cached prompt rates are not sustained-prefill benchmarks.
No universal speed guarantee, isolated generation-patch gain or ComfyUI validation
follows. Shapes, tolerances, hashes and settings are preserved in the JSON.
Manual reproduction uses `tools/benchmark_blobs.py`; `--suite diffusion` selects
the FP32 shapes. Install.cmd never runs that probe. Raw logs and models are not published.
