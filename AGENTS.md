# Project maintenance

- This repository contains only public source, reproducible profiles and sanitized numeric measurements.
- Never commit or upload local runtimes, models, caches, logs, personal paths, credentials or authentication files.
- Do not change GPU drivers, registry, firmware, TDR or security settings.
- Do not run generation, GPU tests or new benchmarks unless the human explicitly requests them.
- Target AMD Polaris / gfx803 / RX580 only. Do not add unmeasured GPU-family profiles.
- Never invent per-patch gains or universal tokens/s. Numerical correctness checks are not speed measurements.
- ComfyUI was installed but not launched in the original experiment. No ComfyUI image-generation speed claim is supported.
- Release archives contain this project's source installer only. Third-party software is downloaded from its publisher.
- Preserve GPL notices and upstream license/source links. Do not upload the original private portable archive.

- Small-GEMM is opt-in and bounded to tiny FP32 on the tested device/context. Preserve the original backend hash and all original export ordinals.
- Historical source-machine results and current-machine checks must stay separate. Never promote generated shim, hsaco, import DLLs or raw reports into Git/source archives.
- Keep Torch DirectML in its separate cp311 environment; DXGI and Torch device indices are not interchangeable.
