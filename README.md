# Polaris AI Toolkit · RX 580 on Windows

[Русский](#русский) · [Source ZIP / Скачать](https://github.com/aktalascloud/polaris-gfx803-rx580-zluda-rocm-hip-comfyui-windows/archive/refs/heads/main.zip) · [Guide / Документация](#установка--setup) · [Evidence / Данные](tests/results.json) · [Patches and licenses](patches/README.md)

Install a pinned AI stack for AMD Polaris / gfx803 using ZLUDA, HIP/ROCm and rocBLAS, with alternate DirectML and OpenVINO CPU backends. ComfyUI is optional. Setup uses the existing compatible AMD driver.

**Quick start.** Download the source ZIP, extract it to a short writable ASCII path and run `Install.cmd`. The installer fetches pinned dependencies from their publishers.

**Compatibility.** On RX 580, ZLUDA 3.9.6 ROCm5 with Torch 2.2.1+cu118 and the matched rocBLAS bundle passed a CUDA kernel check and FP32/FP16 GEMM against CPU. ComfyUI 0.13.0 carries compatibility fixes for that Torch version. See [issue #2](https://github.com/aktalascloud/polaris-gfx803-rx580-zluda-rocm-hip-comfyui-windows/issues/2) and [the test notes](patches/README.md#issue-2--kernel-launch-and-gemm).

## Русский

Установщик закреплённого AI-стека для AMD Polaris / gfx803. В комплекте ZLUDA, HIP/ROCm и rocBLAS, а также DirectML и OpenVINO CPU. ComfyUI подключается по выбору. Используется установленный совместимый драйвер AMD.

**Запуск.** Скачайте ZIP с исходниками, распакуйте его в короткий доступный для записи путь ASCII и запустите `Install.cmd`. Установщик скачает закреплённые зависимости у их издателей.

**Совместимость.** На RX 580 связка ZLUDA 3.9.6 ROCm5, Torch 2.2.1+cu118 и согласованного комплекта rocBLAS прошла проверку CUDA и сверку FP32/FP16 GEMM с CPU. Для этого Torch в ComfyUI 0.13.0 внесены правки совместимости.

## Установка / Setup

Распакуйте ZIP в короткий ASCII-путь, запустите **Install.cmd**, выберите ComfyUI (Enter — да) и подтвердите условия компонентов.

Extract the ZIP to a short writable ASCII path and run **Install.cmd**. Choose ComfyUI (Enter defaults to yes) and accept the dependency terms.

~~~powershell
.\Install.cmd -InstallRoot C:\AI\Polaris-New -WithComfyUI -WithTorchDirectML -WithSmallGemmPatch
~~~

Нужны Windows x64, совместимый драйвер AMD, интернет и около 20 ГиБ. Установка скачивает закреплённые зависимости в новую папку. [Модели](https://huggingface.co/models) скачиваются отдельно.

Setup needs Windows x64, an existing compatible AMD driver, internet and about 20 GiB. It fetches pinned dependencies into a new folder. Download models separately.

| Installed command | Назначение / Purpose |
|---|---|
| Python-CUDA.cmd | PyTorch 2.2.1+cu118, ZLUDA 3.9.6 ROCm5, HIP SDK 5.7.1 |
| Python-ONNX.cmd | ONNX DirectML and OpenVINO CPU |
| Python-DirectML.cmd | Optional isolated Torch DirectML, exposes dml_device |
| Start-ComfyUI.cmd | Optional local UI/API at 127.0.0.1:8188, CPU VAE and automatic bounded attention |
| Check-Zluda.cmd | CUDA/PTX smoke check. Use `--torch` for a PyTorch kernel and FP32/FP16 GEMM check against CPU |
| Run-ZLUDA.cmd, HIP-Shell.cmd | Generic application launcher and process-local HIP shell |

## RX580 results

- **Attention** · 146 synthetic GPU cases passed in each of two processes. FP32 4096×4096 attention: 87.68 → 53.86 ms (−38.6%), additional peak tensor allocations 1024 → 69 MiB.
- **Small-GEMM** · opt-in tiny FP32 correctness fix. Strided baseline 63/69; the validated patch build passed 69/69 on each of four APIs. The v0.4.3 compiler flags were checked at build time; no new GPU results.
- **Backends** · CUDA vector addition 64/64. Torch and strict ONNX DirectML passed. OpenVINO CPU passed, OpenVINO GPU returned incorrect zeros.

Замеры attention и Small-GEMM относятся к прежним сборкам. / Attention and Small-GEMM figures come from earlier builds.

## Структура / Source layout

| Path | Назначение / Purpose |
|---|---|
| [tools/](tools/) | Installer, runtime helpers, source packaging |
| [profiles/](profiles/) | Pinned downloads, Python requirements, source ZIP allowlist |
| [launchers/](launchers/) | Commands copied to the installed runtime |
| [patches/](patches/) | ComfyUI and Small-GEMM changes, rollback and licenses |
| [tests/](tests/) | Checks, reproduction scripts and recorded results |

Local development and verification: [AGENTS.md](AGENTS.md). Source license: [GPL-3.0-only](LICENSE).

[All projects / Все проекты](https://github.com/aktalascloud)
