# Polaris AI Toolkit

**RX580 · gfx803 · Windows x64** — ZLUDA, HIP/ROCm, DirectML and OpenVINO, with optional ComfyUI.

[Русский](README.ru.md) · [Download v0.3.1](https://github.com/aktalascloud/polaris-gfx803-rx580-zluda-rocm-hip-comfyui-windows/releases/tag/v0.3.1) · [Patches](docs/PATCHES.md) · [Licenses](THIRD_PARTY_NOTICES.md)

## Install

Extract the setup ZIP into a short writable ASCII path, such as `C:\AI\Polaris-Setup`, and run **Install.cmd**. Accept the linked vendor terms and choose ComfyUI (`Y` or Enter = yes).

For all optional features:

```powershell
.\Install.cmd -InstallRoot C:\AI\Polaris-New -WithComfyUI -WithTorchDirectML -WithSmallGemmPatch
```

Requires Windows x64, internet, about 20 GiB setup space and an existing compatible AMD driver. Setup downloads pinned dependencies, applies the matched gfx803 libraries and installs VC++ x64 if missing. It creates a ComfyUI desktop shortcut when selected. Existing destinations are refused; the GPU driver is never replaced. Models are separate, licensed downloads from [Hugging Face](https://huggingface.co/models).

| Installed launcher | Purpose |
|---|---|
| `Python-CUDA.cmd` | Torch 2.2.1+cu118 through ZLUDA 3.9.5 / HIP 5.7.1 |
| `Python-ONNX.cmd` | ONNX DirectML 1.24.4 and OpenVINO 2026.4.1; select providers in your code |
| `Python-DirectML.cmd` | Optional isolated Torch DirectML; exposes `dml_device` |
| `Start-ComfyUI.cmd` | Optional local UI/API at `127.0.0.1:8188`; FP32, low VRAM, CPU VAE |
| `Check-Zluda.cmd` | Manual CUDA/PTX check without Torch or models |
| `Run-ZLUDA.cmd`, `HIP-Shell.cmd` | Generic application launcher and process-local HIP shell |

Use launchers inside the installed runtime. CUDA/DirectML wrappers accept `-c`, `-m` and script paths. External API nodes are disabled in the ComfyUI preset; no cloud keys are configured.

## Patches and results

The default uses the matched **advanced-lvl-up rocBLAS DLL + gfx803 kernels**, Torch DLL aliases and preserved original NVRTC. Stock HIP SDK and pinned upstream ZLUDA are not advertised as entirely rewritten/patched. Rejected optic/vega libraries are not installed.

**Small-GEMM is opt-in:** `-WithSmallGemmPatch` fixes tiny FP32 cases (dimensions 1–8) on tested Polaris device 0. Baseline strided coverage is **63/69**; the patch passes **69/69 on each of four APIs**, max error `3.34e-6`; original export ordinals and rollback are verified. This is a correctness fix, not a generation-speed claim.

Torch DirectML FP32/FP16, strict ONNX DirectML, OpenVINO CPU and CUDA vector addition passed on the reference RX580. The installer was checked in a fresh directory with an existing SDK. **ComfyUI startup/image generation remains unverified.** OpenVINO Ellesmere GPU produced incorrect zeros; use CPU.

| Reference | Details |
|---|---|
| [Small-GEMM guide](docs/SMALL-GEMM.md) | Build, install, rollback, runtime selection and limits |
| [Blob comparison](docs/BLOB-RETRY-2026-10-23.md) | Failed candidates, GEMM timings and controlled/practical Vulkan comparisons |
| [Numeric evidence](data/small-gemm-validation-20261023.json) | Current-machine results; source-machine evidence is separate |

No universal tokens/s or images/s gain is claimed. Other drivers/runtimes, real multi-GPU mapping, concurrent native calls, full context reset and native Unicode paths are unverified. The pinned ComfyUI snapshot does not support Z-Image.

The source-only release contains no vendor binaries, models, credentials, private configuration, raw logs or caches. Third-party components keep their own terms; see the AMD/NVIDIA boundaries in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Project code: **GPL-3.0-only**.
