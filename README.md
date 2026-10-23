# Pascal & RX580 AI on Windows — AVX2 · KoboldCPP · ZLUDA · ComfyUI

Pinned Windows recipes for **AMD RX 580 / Polaris gfx803** using ZLUDA/HIP and an
**experimental, unmeasured NVIDIA Pascal** native-CUDA profile. Includes ComfyUI,
KoboldCPP, ONNX Runtime DirectML and OpenVINO CPU.

[Русское описание](README.ru.md) · [Patch catalogue](docs/PATCHES.md) ·
[Third-party terms and source links](THIRD_PARTY_NOTICES.md)

## Optimized KoboldCPP: AVX2 / Vulkan / NUMA

For the author's separately optimized CPU/Vulkan build, see
[KoboldCpp AVX2 / Vulkan / NUMA](https://github.com/aktalascloud/koboldcpp-avx2-vulkan-numa).
That project contains Haswell AVX2 PGO/LTO work, RX580 Vulkan tuning and optional
HybridMoE / Windows NUMA support, with its own releases and measurements.
Its CPU/Vulkan results are separate from this repository's historical ZLUDA data;
the percentages must not be combined or presented as the same experiment.

Search terms: NVIDIA Pascal, AMD Radeon RX580 / RX 580, Polaris gfx803, AVX2,
Kobold / KoboldCPP, ZLUDA, ComfyUI, HIP / ROCm, CUDA 11.8, DirectML, OpenVINO.
AVX2 describes the reference CPU capability and the related optimized fork;
no isolated AVX2 patch gain was measured in this repository.

## What is actually verified

Historical measurements were collected on **2026-10-22**, RX 580 2048SP, gfx803,
8 GiB VRAM, Xeon E5-2696 v3, 32 GiB RAM, Windows 11, AMD driver 31.0.14037.18001.
The driver was preserved. This project does not install or update GPU drivers.

| Component | Evidence/status |
|---|---|
| KoboldCPP 1.95.1 + ZLUDA 3.9.5 + HIP 5.7.1 | Correct control answers and four identical CPU/GPU continuations; 24/37 GPU layers |
| PyTorch 2.2.1+cu118 | FP32/FP16 GEMM, a small convolution network, 60 BMM/copy iterations, LayerNorm/Softmax passed earlier |
| DirectML 1.24.4 | Physical Radeon inference passed earlier, CPU fallback disabled |
| OpenVINO 2026.4.1 | CPU passed earlier; **Ellesmere GPU returned incorrect zeros** |
| ComfyUI-Zluda, commit `c0c011dca91b595091a4b032770da366fc90fb2b` | Installed only; **not launched or image-validated** |
| NVIDIA Pascal | Native CUDA 11.8 recipe only; **no Pascal measurements** |

**No new model generations or GPU benchmarks were run to publish this project.**
The online installer itself has not undergone a fresh end-to-end installation
or hardware test. Its downloaded Python is 3.12.10; the original machine used
3.12.14. Both use CPython's `cp312` wheel ABI, but this is a documented recipe
variation, not another measured run. Other cards/drivers are not certified.

## Recorded speed and token counts

One diagnostic continuation generated **128 tokens at 8.46 decode tokens/s**:
15.13 s decoding, 17.07 s total request time; prefill 8.77 tokens/s.
This is one bring-up measurement, not a sustained benchmark.

| Paired request | CPU total, s | RX580 total, s | Relative request throughput | Latency reduction |
|---|---:|---:|---:|---:|
| Arithmetic | 5.734 | 5.266 | +8.89% | 8.16% |
| Translation | 7.875 | 5.109 | +54.14% | 35.12% |
| Long context | 53.125 | 50.610 | +4.97% | 4.73% |
| Longer output | 11.563 | 11.343 | +1.94% | 1.90% |

These are **single paired end-to-end timings**, CPU with zero GPU layers versus
24 GPU layers, same Qwen3-4B IQ4_XS model, eight threads, context 1024, BLAS batch
disabled on both. They include prompt processing, caching and request overhead.
Other setup/compilation activity was occurring during bring-up; measurements
were not isolated, repeated or randomized. Percentages are **not decode-rate
gains, isolated patch gains, Vulkan comparisons, or general speed guarantees**.

Formulas: throughput change = `(CPU_seconds / GPU_seconds - 1) * 100`;
latency reduction = `(1 - GPU_seconds / CPU_seconds) * 100`.
Both backends made the same arithmetic error on one comparison (`17*23 → 401`),
so identical continuations demonstrate reproducibility, not factual correctness.
Separate simple controls returned `4`, `63`, and `Paris` correctly.

[Paired numeric data](data/paired-latency.csv) · [Token-count/rate data](data/decode.csv).
The 2–3-token controls reached 12.50–16.39 tokens/s, but are too short to advertise
as throughput. **No per-patch percentage, ComfyUI images/s or Pascal tokens/s was measured.**

## Windows release / install

Download the `legacy-gpu-ai-windows-…-windows-setup.zip` asset from
[Releases](https://github.com/aktalascloud/pascal-rx580-avx2-koboldcpp-zluda-comfyui-windows/releases), extract
it into a short writable path, read the third-party notices, then run `Install.cmd`.

The release is a **small online installer**, not the original private/offline
archive. It downloads pinned components directly from their publishers, verifies
published hashes/signatures, patches a fresh local copy and does not launch ML
applications. A compatible **already installed GPU driver**, internet access,
Windows x64 and the Microsoft VC++ 2015–2022 x64 runtime are prerequisites.
If the latter is missing, use [Microsoft's official installer](https://aka.ms/vs/17/release/vc_redist.x64.exe).
Allow roughly 20 GiB of free space during setup. MSI administrative extraction
may need additional local permissions; it is not a GPU-driver installation.

```powershell
# AMD RX580: download/extract SDK components without running AMD Setup.
.\Install.cmd -Backend AMD -AcceptThirdPartyLicenses

# Optional: use an existing HIP SDK 5.7.1 as the source of a fresh local copy.
.\Install.cmd -Backend AMD -HipRoot "C:\Program Files\AMD\ROCm\5.7" -AcceptThirdPartyLicenses

# NVIDIA Pascal: native CUDA, no ZLUDA DLL replacement or HIP download.
# This path is unmeasured, not a tested Pascal performance claim.
.\Install.cmd -Backend NVIDIA -AcceptThirdPartyLicenses
```

`-AcceptThirdPartyLicenses` is an explicit acknowledgement to review and comply
with the linked terms, not a replacement license or a grant from this project.
Choose a fresh destination with `-InstallRoot`; existing installations are never
overwritten. A failed setup can leave an incomplete folder; preserve/review it
before choosing another destination. No benchmark runs automatically after setup.

Use the launchers **inside the installed `runtime` folder**:

- `Start-ComfyUI.cmd`: local UI on port 8188; FP32/low-VRAM, CPU VAE.
- `Start-Kobold.cmd`: local UI on port 5021; optional `-ModelPath` and `-GpuLayers`.
- `Python-CUDA.cmd script.py`: pinned PyTorch environment.
- `Python-ONNX.cmd script.py`: ONNX Runtime DirectML / OpenVINO environment.

## Models are not distributed

Use `Download-Model.cmd` for the optional, hash-pinned
[Qwen3-4B IQ4_XS model card](https://huggingface.co/mradermacher/Qwen3-4B-Instruct-2507-heretic-uncens-i1-GGUF),
or download your own properly licensed GGUF into `runtime/models`.
The script requests no account credential and downloads **from Hugging Face**,
not this repository or its release. Model weights are never uploaded here.

ComfyUI checkpoints are separate downloads too. Put a compatible, licensed
SD1.x/SDXL checkpoint in `runtime/apps/ComfyUI/models/checkpoints`.
The pinned 2025-08-31 ComfyUI snapshot **does not support Z-Image**. Pointing it at
a directory named ZIMAGE does not add that architecture. There is no automatic
checkpoint download, manager installation or custom-node auto-update.

## Critical workarounds and limits

- RX580 KoboldCPP needs **`--blasbatchsize -1`** and `nommq`. GPU offload remains
  enabled. Normal batching produced incorrect text; this is a correctness workaround.
- Use the matched gfx803 rocBLAS **DLL and its kernel library** together.
- ZLUDA 3.9.5, Torch 2.2.1+cu118 and NumPy 1.26.4 are pinned. Do not `pip install -U torch`.
- Disable cuDNN/TF32/unsupported attention paths on AMD. NVIDIA uses native CUDA.
- Tiny batched GEMM `[3,1,2] × [3,2,1]` has no Tensile solution in this library.
- Wave64 slow mode and MMQ failed earlier. Full 250-convolution Torch inference
  was not confirmed. See the [patch catalogue](docs/PATCHES.md).
- DirectML here means **ONNX Runtime DirectML**, not `torch-directml` or a ComfyUI DML mode.
- OpenVINO must explicitly target **CPU** on this setup.

## Privacy and licensing

Git and the public release contain source scripts, a small patch, fixed profiles,
license notices and sanitized numeric tables only. No local runtime, model,
cache, raw log, system DLL, personal configuration, credential or private path is
published. Downloads occur on the recipient's machine under upstream terms.

Project code: **GPL-3.0-only**. Third-party software retains its own licenses.
In particular, the AMD runtime belongs to the installed GPU driver; NVIDIA SDK
terms also contain translation restrictions. This project does not redistribute
those vendor binaries or declare third-party use automatically licensed.
Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before setup.

Not affiliated with AMD, NVIDIA, ComfyUI, PyTorch or the upstream maintainers.
