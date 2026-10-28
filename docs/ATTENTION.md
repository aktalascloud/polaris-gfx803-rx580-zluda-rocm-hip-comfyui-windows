# Default bounded attention · RX580

Enabled automatically for newly installed ComfyUI with the pinned source commit.
**No additional installer option.** Existing working installations are not modified.
The idea comes from advanced-lvl-up's [GPL jab-attention helper](https://github.com/advanced-lvl-up/Rx470-Vega10-Rx580-gfx803-gfx900-fix-AMD-GPU/blob/6547664db8732305530b969b54c9b7d3806c02e1/jab-attention-VRAM-helper.zip);
this adaptation keeps the original API instead of replacing the whole upstream module.

## Contract and dispatch

- Boolean masks: `True` means attend. Floating masks are additive biases; `-inf` blocks a key. Fully masked rows are zero for the finite inputs in the suite.
- Mask shapes: `[K]`, `[Q,K]`, `[B,Q,K]`, or broadcastable `[B,H,Q,K]`. Use `[B,1,1,K]` for batched padding. No full broadcast-mask copy is created.
- Input layouts: `[B,Q,H*D]` or `[B,H,Q,D]` with `skip_reshape=True`. Output is `[B,Q,H*Dv]` or `[B,H,Q,Dv]` with `skip_output_reshape=True`; input dtype is retained.
- FP16 inputs compute internally in FP32 because the legacy library lacks some sliced half GEMMs. Upcast occurs before matmul/softmax/value projection and remains effective inside autocast.
- Small ordinary FP32 calls keep upstream attention. Large calls (estimated score ≥8 MiB), masked calls and FP16 inference use bounded head/query tiles. The score budget is at most 32 MiB and adapts to available memory; it is not a bound on total VRAM.
- Explicit attention backends, CPU/unsupported dtypes, training and explicit no-upcast-half behavior remain upstream. Very long keys that cannot fit one score row retain upstream KV chunking. Training correctness is not established by selector unit tests.

## Recorded checks

**146 GPU cases in each of two processes**, 53 CPU cases, four negative guards and
seven selector checks passed. They include boolean/additive/broadcast masks,
fully blocked rows, noncontiguous inputs, layout combinations, tail tiles, unequal
value depth, empty axes, keyword arguments and overflow-sensitive FP16 upcast.
Reference: independent CPU FP64 attention; fixed tolerances FP32 `atol=3e-5,
rtol=1e-4`, FP16 `atol=0.003, rtol=0.002`.

FP32 timings compare the installed default dispatcher with pinned ComfyUI math.
Medians of 18 alternating warm samples/algorithm across two processes:

| [B,H,Q,K,D] | Stock, ms | Default, ms | Latency change | Extra tensor peak, MiB |
|---|---:|---:|---:|---:|
| [1, 8, 77, 77, 64] | 0.771 | 0.773 | +0.3% | 0.362305 / 0.362305 |
| [1, 8, 256, 77, 64] | 0.804 | 0.806 | +0.2% | 1.20312 / 1.20312 |
| [1, 8, 512, 512, 64] | 2.770 | 2.089 | -24.6% | 16 / 9 |
| [1, 8, 1024, 1024, 64] | 7.113 | 6.385 | -10.2% | 64 / 18 |
| [1, 8, 4096, 4096, 40] | 87.683 | 53.858 | -38.6% | 1024 / 69 |
| [1, 3, 513, 1025, 32] | 2.535 | 2.472 | -2.5% | 12.0361 / 12.0361 |
| [1, 8, 1024, 77, 64] | 0.998 | 0.989 | -0.9% | 4.8125 / 4.8125 |

Memory means **additional peak tensor allocations**, excluding existing inputs,
weights, CUDA/driver allocations and reserved cache. Small stock calls can have
minor dispatch overhead. Larger gains do not establish an overall images/s or
video-generation improvement. No models were downloaded; no generation occurred.
[All cases, samples and limitations](../data/attention-validation-20261028.json).

## Install, inspect and restore

```powershell
.\Install.cmd -WithComfyUI -InstallRoot C:\AI\Polaris-New
C:\AI\Polaris-New\envs\diagnostics\python.exe -I tools\attention_patch.py status C:\AI\Polaris-New\apps\ComfyUI
# Close consumers before restoring the pinned original source:
C:\AI\Polaris-New\envs\diagnostics\python.exe -I tools\attention_patch.py rollback C:\AI\Polaris-New\apps\ComfyUI
```

The real installer ComfyUI source block, idempotent apply, rollback and reapply
were checked on a fresh pinned source copy without starting its server. Unknown
source revisions/modifications are refused. The original source remains the
verified prefix of the patched file; no separate backup is needed. A disabled
helper/receipt can remain for inspection. Diagnostics report hook/hash status.

The full dependency/SDK installer was not repeated for this change; its unchanged
bootstrap was validated earlier. Complete ComfyUI startup/workflows, model quality,
training, other hardware, global VRAM-pressure scenarios and full-model performance
remain unverified. The working RX580 backend, ComfyUI attention and launcher hashes
were unchanged. Source only: no DLLs, weights, raw logs, private paths or caches.
