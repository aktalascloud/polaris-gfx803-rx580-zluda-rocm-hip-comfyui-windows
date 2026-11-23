# SPDX-License-Identifier: GPL-3.0-only
"""Run with Python-CUDA.cmd to check CUDA kernels and rocBLAS against CPU."""
import json

import torch


def main():
    print(f'Torch {torch.__version__}', flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/ZLUDA is unavailable')
    print('Checking CUDA elementwise and reduction', flush=True)
    host = torch.arange(10, dtype=torch.float32)
    result = (host.cuda() * 2).sum().item()
    if result != 90.0:
        raise RuntimeError(f'CUDA elementwise/reduction returned {result}, expected 90')

    rng = torch.Generator().manual_seed(580)
    a = torch.randn(256, 256, generator=rng)
    b = torch.randn(256, 256, generator=rng)
    errors = {}
    for dtype, rtol, atol in [(torch.float32, 0.0002, 0.0001), (torch.float16, 0.002, 0.02)]:
        print(f'Checking 256x256 rocBLAS GEMM ({dtype})', flush=True)
        left, right = a.to(dtype), b.to(dtype)
        expected = left.float() @ right.float()
        actual = (left.cuda() @ right.cuda()).cpu().float()
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        errors[str(dtype)] = (actual - expected).abs().max().item()

    print(json.dumps({'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                      'sum': result, 'gemm_max_abs_error': errors, 'passed': True}), flush=True)


if __name__ == '__main__':
    main()
