# SPDX-License-Identifier: GPL-3.0-only
"""Configure only this process; never change a system driver or global PATH."""
import ctypes
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
from hip_runtime import resolve

ROOT=Path(__file__).resolve().parent.parent
HANDLES=[]
ROCBLAS_DLL_SHA256 = '9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d'
ROCBLAS_LIBRARY_SHA256 = '8c29c48aef06001515e1df959779070c6e2e1c85c71dc2ce667ed4cdc09da107'


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256')


def verify_rocblas(bin_dir, *, library=None):
    if library is None:
        library = bin_dir/'rocblas'/'library'
    receipt = bin_dir/'small-gemm-patch.json'
    patch = json.loads(receipt.read_text(encoding='utf-8-sig')) if receipt.is_file() else {}
    patched = patch.get('enabled', False)
    dll = bin_dir/('rocblas_backend.dll' if patched else 'rocblas.dll')
    if file_digest(dll).hexdigest() != ROCBLAS_DLL_SHA256:
        raise ValueError('Unrecognized rocBLAS DLL. Restore the matched advanced-lvl-up gfx803 bundle.')
    if patched:
        for name in ('rocblas.dll', 'rocblas_small_gemm.hsaco'):
            if file_digest(bin_dir/name).hexdigest() != patch['files'][name]:
                raise ValueError(f'Small-GEMM artifact changed: {name}')
    tree = hashlib.sha256()
    for path in sorted(library.iterdir(), key=lambda file: file.name):
        tree.update(path.name.encode('utf-8') + b'\0' + file_digest(path).digest())
    if tree.hexdigest() != ROCBLAS_LIBRARY_SHA256:
        raise ValueError('rocBLAS library does not match its DLL. Restore the entire pinned gfx803 bundle.')
    return patched

def configure():
    backend=json.loads((ROOT/'profile.json').read_text(encoding='utf-8-sig'))['backend']
    if backend!='AMD':
        raise RuntimeError('This package targets AMD Polaris/gfx803 only')
    if backend=='AMD':
        hip=ROOT/'hip'/'bin'
        zluda=ROOT/'zluda'
        os.environ['HIP_PATH']=str(ROOT/'hip')
        os.environ['ROCBLAS_TENSILE_LIBPATH']=str(hip/'rocblas'/'library')
        runtime=resolve(ROOT)
        for path in (runtime.parent,hip,zluda):
            HANDLES.append(os.add_dll_directory(str(path)))
        # The HIP runtime belongs to the installed AMD GPU driver, not this release.
        HANDLES.append(ctypes.WinDLL(str(runtime)))
        # Forwarding shim requires its original backend in the same folder.
        if verify_rocblas(hip):
            HANDLES.append(ctypes.WinDLL(str(hip/'rocblas_backend.dll')))
        HANDLES.append(ctypes.WinDLL(str(hip/'rocblas.dll')))
        HANDLES.append(ctypes.WinDLL(str(zluda/'nvcuda.dll')))
    os.environ['DISABLE_ADDMM_CUDA_LT']='1'
    return backend

def require_pinned_torch():
    installed = version('torch')
    if installed != '2.2.1+cu118':
        raise RuntimeError(f'Polaris ZLUDA requires torch 2.2.1+cu118 here (found {installed}). '
                           'Restore the pinned CUDA environment and run Check-Zluda.cmd --torch.')


def torch_profile():
    require_pinned_torch()
    backend=configure()
    import torch
    torch.backends.cudnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if backend=='AMD' and hasattr(torch.backends.cuda,'preferred_blas_library'):
        torch.backends.cuda.preferred_blas_library('cublas')
    return torch
