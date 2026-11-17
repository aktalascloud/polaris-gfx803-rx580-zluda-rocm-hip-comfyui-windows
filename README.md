# Polaris AI Toolkit

RX580 / gfx803 · Windows x64 · ZLUDA, HIP/ROCm, DirectML, OpenVINO and optional ComfyUI.

[Download v0.4.2](https://github.com/aktalascloud/polaris-gfx803-rx580-zluda-rocm-hip-comfyui-windows/releases/tag/v0.4.2) · [Patches, rollback and licenses](patches/README.md) · [Numeric results](results.json)

## Установка / Setup

Распакуйте ZIP в короткий ASCII-путь и запустите **Install.cmd**. Выберите ComfyUI (Enter — да) и подтвердите условия компонентов. Для установки с параметрами откройте PowerShell в папке распакованного ZIP.

Extract the ZIP to a short writable ASCII path and run **Install.cmd**. Choose ComfyUI (Enter defaults to yes) and accept the dependency terms.

~~~powershell
.\Install.cmd -InstallRoot C:\AI\Polaris-New -WithComfyUI -WithTorchDirectML -WithSmallGemmPatch
~~~

Нужны Windows x64, совместимый установленный драйвер AMD, интернет и около 20 ГиБ. Установщик скачивает закреплённые зависимости, не перезаписывает существующую папку и не меняет драйвер. Модели скачиваются отдельно с [Hugging Face](https://huggingface.co/models). В исходном архиве код и обезличенные замеры.

Setup requires Windows x64, an existing compatible AMD driver, internet and about 20 GiB. It fetches pinned dependencies and refuses existing destinations. Models are separate downloads. The source ZIP contains project code and sanitized measurements.

| Installed command | Назначение / Purpose |
|---|---|
| Python-CUDA.cmd | PyTorch 2.2.1+cu118, ZLUDA 3.9.5, HIP 5.7.1 |
| Python-ONNX.cmd | ONNX DirectML and OpenVINO CPU |
| Python-DirectML.cmd | Optional isolated Torch DirectML, exposes dml_device |
| Start-ComfyUI.cmd | Optional local UI/API at 127.0.0.1:8188, CPU VAE and automatic bounded attention |
| Check-Zluda.cmd | Manual CUDA/PTX check without models |
| Run-ZLUDA.cmd, HIP-Shell.cmd | Generic application launcher and process-local HIP shell |

## RX580 results

- **Attention** · 146 synthetic GPU cases passed in each of two processes. FP32 4096×4096 attention: 87.68 → 53.86 ms (−38.6%), additional peak tensor allocations 1024 → 69 MiB.
- **Small-GEMM** · opt-in tiny FP32 correctness fix. Strided baseline 63/69. Patched 69/69 on each of four APIs.
- **Backends** · CUDA vector addition 64/64. Torch and strict ONNX DirectML passed. OpenVINO CPU passed, OpenVINO GPU returned incorrect zeros.

Замеры относятся к указанным операциям. Полный запуск ComfyUI, генерация и её скорость не проверены. [Методика, ограничения и откат](patches/README.md).

The recorded figures cover individual operations. Full ComfyUI startup and generation remain unverified. Project source is [GPL-3.0-only](LICENSE).
