# Polaris / gfx803 / RX580: ZLUDA, HIP/ROCm, DirectML and OpenVINO for Windows

A general-purpose local AI backend toolkit, with optional ComfyUI-Zluda.
[Русский](README.ru.md) · [Patches](docs/PATCHES.md) · [Licenses](THIRD_PARTY_NOTICES.md)

## Install

1. Download the setup ZIP from [Releases](https://github.com/aktalascloud/polaris-gfx803-rx580-zluda-rocm-hip-comfyui-windows/releases/tag/v0.2.1), extract into a short writable folder such as `C:\AI\Polaris-Setup`.
2. Double-click **Install.cmd**, choose whether to include ComfyUI (Enter = yes), and accept the linked vendor terms.
3. Setup downloads and installs all pinned Python packages, ZLUDA, HIP SDK, patched rocBLAS and matching gfx803 kernel blobs automatically. It installs the signed Microsoft VC++ x64 runtime if missing; Windows may request administrator consent.
4. If selected, ComfyUI receives its dependencies and startup patch automatically, plus a **ComfyUI - Polaris ZLUDA** desktop shortcut. It is not started during setup.

Internet access, Windows x64, approximately 20 GiB temporary free space and an **existing compatible AMD driver containing amdhip64.dll** are required. Setup never installs or replaces a GPU driver, changes TDR, registry or firmware. It refuses an existing destination to protect installations. A failed attempt can leave a partial folder; select a fresh destination on retry.

```powershell
# Explicit selections; vendor terms must first be reviewed.
.\Install.cmd -WithComfyUI -AcceptThirdPartyLicenses
.\Install.cmd -WithoutComfyUI -AcceptThirdPartyLicenses
# Optional: -InstallRoot C:\AI\Polaris-Runtime -HipRoot "C:\Program Files\AMD\ROCm\5.7" -NoDesktopShortcut
```

Default destination: `runtime` next to Install.cmd. Use launchers inside that installed folder:

- `Python-CUDA.cmd script.py`: PyTorch 2.2.1+cu118 through ZLUDA, with local DLL loading and conservative Torch settings.
- `Python-ONNX.cmd script.py`: separate ONNX Runtime DirectML 1.24.4 / OpenVINO 2026.4.1 environment. Select `DmlExecutionProvider` explicitly for DirectML; select `CPU` for OpenVINO. Ellesmere OpenVINO GPU produced incorrect zeros earlier.
- `Run-ZLUDA.cmd program.exe [arguments]`: run another compatible CUDA application with the local HIP/ZLUDA search path. Applications may require their own DLL aliases and dependencies; arbitrary CUDA programs are not certified.
- `HIP-Shell.cmd`: command prompt with process-local HIP paths, suitable for your own HIP tools.
- `Start-ComfyUI.cmd`, when selected: local UI/API at `http://127.0.0.1:8188`, FP32, low VRAM and CPU VAE. External API nodes are disabled; no API keys or cloud service accounts are configured.

Models are not included or downloaded automatically. Obtain appropriately licensed checkpoints from [Hugging Face](https://huggingface.co/models), following the selected application's format and model-folder instructions. This toolkit does not require a specific language-model application.

## Patches are applied automatically

The **installed toolkit** already has the required gfx803 overrides: advanced-lvl-up rocBLAS DLL **and its matching kernel blobs**, pinned ZLUDA 3.9.5, Torch DLL aliases, preserved original NVRTC and process-local loading. No manual blob copying is needed after successful setup.

This does **not** mean stock HIP 5.7.1 or every ROCm component was independently patched. The Windows SDK is copied from AMD's original package, then its rocBLAS DLL/kernel directory is replaced. ZLUDA is the pinned upstream build, not a newly modified binary. Linux Schaka patches are research references, not installed Windows patches.

The retained blobs come from [advanced-lvl-up v1.0.0](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/releases/tag/v1.0.0). Earlier likelovewant v0.5.7 candidates were rejected. The fresh retry below found that optic can pass selected checks but remains unstable and is excluded from automatic installation. Per-blob speedup was not measured; compatibility/correctness is supported by historical checks, not an isolated acceleration claim.

The repository includes the ComfyUI source patch, complete install logic, pinned publisher URLs and SHA256 values in [components.json](profiles/components.json). The release is a source-only **online installer**: binaries are fetched automatically from their publishers. The rocBLAS patch publisher has no explicit repository license; its source archive also lacks a top-level license, so its patched binary is not mirrored here. Upstream rocBLAS licensing alone does not establish permission for every third-party modification.

## Fresh blob retry: 2026-10-23

[Full methods, patch decisions and tables](docs/BLOB-RETRY-2026-10-23.md) ·
[Sanitized numeric data](data/blob-retry-2026-10-23.json).

- Baseline matched rocBLAS/blobs passed both eight-check FP32/FP16 runs.
- Full likelovewant **optic** passed once, then crashed on repeat before FP16. Its 24-layer application decode median was **8.10 tokens/s**, versus baseline **8.25 tokens/s**: **-1.78%**, not a twofold improvement.
- On four diffusion-like FP32 GEMM shapes, optic throughput changed **-54.19%, -66.11%, -66.46%, -0.05%**. These are matrix probes, not ComfyUI images/s.
- Full vega fallback crashed. All four partial DLL/kernel swaps crashed or failed correctness; they are not installation options.
- Same oldpc application build / 24 GPU layers: Vulkan **9.58 tokens/s** without BLAS batching; **9.25 tokens/s** with batch 512 and substantially faster prompt processing. ZLUDA batching 512 with optic crashed before controls.
- The separate optimized Vulkan configuration reached **33.09 tokens/s** versus ZLUDA **8.25 tokens/s**. This also changes version, offload count, CPU threads and custom kernels; its roughly **4.01x** ratio is a configuration comparison, not a rocBLAS patch gain. Samples were two 128-token requests for optimized Vulkan, three for the 24-layer reference.

**Default remains the matched advanced-lvl-up pair.** No unstable blob package or mixed set is promoted. Existing working files and protected driver files were unchanged. The source-only installer and optional ComfyUI generation have not received fresh end-to-end validation. GPU workload scripts are manual only; Install.cmd never runs them.

## Historical verification and limits

Recorded on 2026-10-22: RX580 2048SP, Polaris/gfx803, 8 GiB, Windows 11, existing AMD driver 31.0.14037.18001. PyTorch FP32/FP16 GEMM, a small convolutional network, 60 BMM/copy iterations and LayerNorm/Softmax passed earlier. Tiny batched GEMM still has unsupported shapes. Physical Radeon DirectML passed with CPU fallback disabled; OpenVINO CPU passed.

ComfyUI commit `c0c011dca91b595091a4b032770da366fc90fb2b` was installed previously but **not launched or validated by image generation**; this legacy snapshot does not provide Z-Image support. Fresh isolated GEMM and application tests were explicitly authorized on 2026-10-23; see the linked retry report. The revised installer has only static checks, not a fresh end-to-end install. Downloaded Python 3.12.10 differs from the original host's 3.12.14, sharing cp312 ABI.

No universal tokens/s, images/s or per-patch percentage is claimed. Application-specific launchers and model downloaders are not part of this general toolkit. The documented language-model application is a measurement fixture only.

Project source: GPL-3.0-only. Third-party software keeps its own terms. No models, secrets, personal configuration, logs, caches or system DLLs are published. Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including AMD and NVIDIA component boundaries.
