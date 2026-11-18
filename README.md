# Polaris AI Toolkit · RX 580 on Windows

[Русский](#русский) · [Release / Скачать](https://github.com/aktalascloud/polaris-gfx803-rx580-zluda-rocm-hip-comfyui-windows/releases/tag/v0.4.3) · [Guide / Документация](#установка--setup) · [Evidence / Данные](tests/results.json) · [Patches and licenses](patches/README.md)

Install a pinned AI stack for AMD Polaris / gfx803 using ZLUDA, HIP/ROCm and rocBLAS, with alternate DirectML and OpenVINO CPU backends. ComfyUI is optional. Setup uses the existing compatible AMD driver.

**Quick start.** Download the release source ZIP, extract it to a short writable ASCII path and run `Install.cmd`. The installer fetches pinned dependencies from their publishers.

**Evidence.** Earlier RX 580 synthetic FP32 4096×4096 attention runs measured 87.68 → 53.86 ms and 1,024 → 69 MiB in additional peak tensor allocations. The Small-GEMM v0.4.3 prerelease verified the gfx803 build and all 3,936 export ordinals. The speed figures refer to the earlier attention experiment.

## Русский

Установщик закреплённого AI-стека для AMD Polaris / gfx803. В комплекте ZLUDA, HIP/ROCm и rocBLAS, а также DirectML и OpenVINO CPU. ComfyUI подключается по выбору. Используется установленный совместимый драйвер AMD.

**Запуск.** Скачайте ZIP с исходниками, распакуйте его в короткий доступный для записи путь ASCII и запустите `Install.cmd`. Установщик скачает закреплённые зависимости у их издателей.

**Проверка.** В более раннем синтетическом тесте FP32 attention 4096×4096 на RX 580 получено 87,68 → 53,86 мс и 1 024 → 69 МиБ дополнительных пиковых выделений тензоров. В v0.4.3 проверены сборка gfx803 и все 3 936 экспортных ординалов Small-GEMM. Замер скорости относится к прежнему тесту attention.

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
| Python-CUDA.cmd | PyTorch 2.2.1+cu118, ZLUDA 3.9.5, HIP 5.7.1 |
| Python-ONNX.cmd | ONNX DirectML and OpenVINO CPU |
| Python-DirectML.cmd | Optional isolated Torch DirectML, exposes dml_device |
| Start-ComfyUI.cmd | Optional local UI/API at 127.0.0.1:8188, CPU VAE and automatic bounded attention |
| Check-Zluda.cmd | Manual CUDA/PTX check without models |
| Run-ZLUDA.cmd, HIP-Shell.cmd | Generic application launcher and process-local HIP shell |

## RX580 results

- **Attention** · 146 synthetic GPU cases passed in each of two processes. FP32 4096×4096 attention: 87.68 → 53.86 ms (−38.6%), additional peak tensor allocations 1024 → 69 MiB.
- **Small-GEMM** · opt-in tiny FP32 correctness fix. Strided baseline 63/69; the validated patch build passed 69/69 on each of four APIs. The v0.4.3 compiler flags were checked at build time; no new GPU results.
- **Backends** · CUDA vector addition 64/64. Torch and strict ONNX DirectML passed. OpenVINO CPU passed, OpenVINO GPU returned incorrect zeros.

Полный запуск и генерация ComfyUI не проверены. / Full ComfyUI startup and generation remain unverified.

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
