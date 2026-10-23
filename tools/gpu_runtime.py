# SPDX-License-Identifier: GPL-3.0-only
"""Configure only this process; never change a system driver or global PATH."""
import ctypes
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
HANDLES=[]

def configure():
    backend=json.loads((ROOT/'profile.json').read_text(encoding='utf-8-sig'))['backend']
    if backend!='AMD':
        raise RuntimeError('This package targets AMD Polaris/gfx803 only')
    if backend=='AMD':
        hip=ROOT/'hip'/'bin'
        zluda=ROOT/'zluda'
        os.environ['HIP_PATH']=str(ROOT/'hip')
        os.environ['ROCBLAS_TENSILE_LIBPATH']=str(hip/'rocblas'/'library')
        for path in (hip,zluda):
            HANDLES.append(os.add_dll_directory(str(path)))
        # The HIP runtime belongs to the installed AMD GPU driver, not this release.
        HANDLES.append(ctypes.WinDLL(str(Path(os.environ['SystemRoot'])/'System32'/'amdhip64.dll')))
        HANDLES.append(ctypes.WinDLL(str(hip/'rocblas.dll')))
        HANDLES.append(ctypes.WinDLL(str(zluda/'nvcuda.dll')))
    os.environ['DISABLE_ADDMM_CUDA_LT']='1'
    return backend

def torch_profile():
    backend=configure()
    import torch
    torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if backend=='AMD' and hasattr(torch.backends.cuda,'preferred_blas_library'):
        torch.backends.cuda.preferred_blas_library('cublas')
    return torch
