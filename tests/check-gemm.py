# SPDX-License-Identifier: GPL-3.0-only
import ctypes as c
import hashlib
import itertools
import json
import os
import argparse
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '4'
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dll', required=True)
parser.add_argument('--hip-root', required=True)
parser.add_argument('--hip-runtime')
parser.add_argument('--allow-unsupported', action='store_true')
parser.add_argument('--report', default='gemm-regression.json')
parser.add_argument('--api', choices=['strided', 'sgemm', 'ex', 'strided-ex'], default='strided')
parser.add_argument('--nondefault-stream', action='store_true')
arguments=parser.parse_args()
hip_bin = Path(arguments.hip_root).resolve() / 'bin'
dll_directory = os.add_dll_directory(str(hip_bin))
runtime = Path(arguments.hip_runtime) if arguments.hip_runtime else hip_bin / 'amdhip64.dll'
if arguments.hip_runtime:
    runtime=runtime.resolve(strict=True)
elif not runtime.exists():
    runtime=Path(os.environ['SystemRoot']) / 'System32' / 'amdhip64.dll'
hip = c.WinDLL(str(runtime.resolve()))
rocblas_path = Path(arguments.dll) if arguments.dll else hip_bin / 'rocblas.dll'
backend_path=rocblas_path.parent / 'rocblas_backend.dll'
backend_handle=c.WinDLL(str(backend_path)) if backend_path.exists() else None
blas = c.WinDLL(str(rocblas_path))


def bind(library, name, types):
    result = getattr(library, name)
    result.argtypes = types
    result.restype = c.c_int
    return result


def check(code, operation):
    if code:
        raise RuntimeError(f'{operation} returned {code}')


allocate = bind(hip, 'hipMalloc', [c.POINTER(c.c_void_p), c.c_size_t])
free = bind(hip, 'hipFree', [c.c_void_p])
copy = bind(hip, 'hipMemcpy', [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int])
synchronize = bind(hip, 'hipDeviceSynchronize', [])
create = bind(blas, 'rocblas_create_handle', [c.POINTER(c.c_void_p)])
destroy = bind(blas, 'rocblas_destroy_handle', [c.c_void_p])
gemm = bind(blas, 'rocblas_sgemm_strided_batched', [c.c_void_p, c.c_int, c.c_int, c.c_int, c.c_int, c.c_int, c.POINTER(c.c_float), c.c_void_p, c.c_int, c.c_int64, c.c_void_p, c.c_int, c.c_int64, c.POINTER(c.c_float), c.c_void_p, c.c_int, c.c_int64, c.c_int])
sgemm=bind(blas,'rocblas_sgemm',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.POINTER(c.c_float),c.c_void_p,c.c_int,c.c_void_p,c.c_int,c.POINTER(c.c_float),c.c_void_p,c.c_int])
gemm_ex=bind(blas,'rocblas_gemm_ex',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.c_uint])
strided_ex=bind(blas,'rocblas_gemm_strided_batched_ex',[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_int64,c.c_void_p,c.c_int,c.c_int,c.c_int64,c.c_void_p,c.c_void_p,c.c_int,c.c_int,c.c_int64,c.c_void_p,c.c_int,c.c_int,c.c_int64,c.c_int,c.c_int,c.c_int,c.c_int,c.c_uint])
handle = c.c_void_p()
check(create(c.byref(handle)), 'rocblas_create_handle')
stream=c.c_void_p()
if arguments.nondefault_stream:
    check(bind(hip,'hipStreamCreateWithFlags',[c.POINTER(c.c_void_p),c.c_uint])(c.byref(stream),1),'hipStreamCreate')
    check(bind(blas,'rocblas_set_stream',[c.c_void_p,c.c_void_p])(handle,stream),'rocblas_set_stream')
random = np.random.default_rng(20261022)
results = []


def run_case(m, n, k, batch, transpose_a, transpose_b, repeats=1):
    allocations = []
    try:
        sizes = [batch * m * k, batch * k * n, batch * m * n]
        if arguments.api in ['ex','strided-ex']:sizes.append(batch*(m+2)*n)
        for size in sizes:
            address = c.c_void_p()
            check(allocate(c.byref(address), size * 4), 'hipMalloc')
            allocations.append(address)
        maximum_error = 0.0
        for _ in range(repeats):
            a = random.uniform(-0.5, 0.5, (batch, m, k)).astype(np.float32)
            b = random.uniform(-0.5, 0.5, (batch, k, n)).astype(np.float32)
            initial_c = random.uniform(-0.5, 0.5, (batch, m, n)).astype(np.float32)
            a_storage = a.transpose(0, 2, 1) if transpose_a else a
            b_storage = b.transpose(0, 2, 1) if transpose_b else b
            buffers = [a_storage.transpose(0, 2, 1).copy(), b_storage.transpose(0, 2, 1).copy(), initial_c.transpose(0, 2, 1).copy()]
            for address, buffer in zip(allocations, buffers):
                check(copy(address, buffer.ctypes.data, buffer.nbytes, 1), 'hipMemcpy host-to-device')
            if arguments.api in ['ex','strided-ex']:
                sentinel=np.full((batch,n,m+2),12345.0,dtype=np.float32)
                check(copy(allocations[3],sentinel.ctypes.data,sentinel.nbytes,1),'initialize separate output')
            alpha, beta = c.c_float(0.75), c.c_float(0.25)
            ta,tb=112 if transpose_a else 111,112 if transpose_b else 111
            lda,ldb=k if transpose_a else m,n if transpose_b else k
            if arguments.api=='strided':
                status=gemm(handle,ta,tb,m,n,k,c.byref(alpha),allocations[0],lda,m*k,allocations[1],ldb,k*n,c.byref(beta),allocations[2],m,m*n,batch)
            elif arguments.api=='strided-ex':
                status=strided_ex(handle,ta,tb,m,n,k,c.byref(alpha),allocations[0],151,lda,m*k,allocations[1],151,ldb,k*n,c.byref(beta),allocations[2],151,m,m*n,allocations[3],151,m+2,(m+2)*n,batch,151,0,0,0)
            else:
                status=0
                for batch_index in range(batch):
                    ap=c.c_void_p(allocations[0].value+batch_index*m*k*4)
                    bp=c.c_void_p(allocations[1].value+batch_index*k*n*4)
                    cp=c.c_void_p(allocations[2].value+batch_index*m*n*4)
                    if arguments.api=='sgemm':status=sgemm(handle,ta,tb,m,n,k,c.byref(alpha),ap,lda,bp,ldb,c.byref(beta),cp,m)
                    else:
                        dp=c.c_void_p(allocations[3].value+batch_index*(m+2)*n*4)
                        status=gemm_ex(handle,ta,tb,m,n,k,c.byref(alpha),ap,151,lda,bp,151,ldb,c.byref(beta),cp,151,m,dp,151,m+2,151,0,0,0)
                    if status:break
            if status:
                results.append({'shape_mnk': [m, n, k], 'batch': batch, 'transpose_a': transpose_a, 'transpose_b': transpose_b, 'repeats': repeats, 'rocblas_status': status, 'supported': False})
                return
            check(synchronize(), 'hipDeviceSynchronize')
            extra_output=arguments.api in ['ex','strided-ex']
            output = np.empty((batch, n, m+2 if extra_output else m), dtype=np.float32)
            check(copy(output.ctypes.data, allocations[3] if extra_output else allocations[2], output.nbytes, 2), 'hipMemcpy device-to-host')
            if extra_output:
                if not np.all(output[:,:,m:]==12345.0):raise RuntimeError('Output padding was overwritten')
                output=output[:,:,:m]
            expected = (0.75 * (a.astype(np.float64) @ b.astype(np.float64)) + 0.25 * initial_c).astype(np.float32)
            actual = output.transpose(0, 2, 1)
            np.testing.assert_allclose(actual, expected, rtol=0.0001, atol=0.00003)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        results.append({'shape_mnk': [m, n, k], 'batch': batch, 'transpose_a': transpose_a, 'transpose_b': transpose_b, 'repeats': repeats, 'max_abs_error': maximum_error, 'supported': True})
    finally:
        for address in allocations:
            free(address)


try:
    for shape, batch, transpose_a, transpose_b in itertools.product([(1, 1, 1), (2, 2, 1), (4, 4, 4), (8, 8, 8), (7, 3, 5), (3, 7, 2), (1, 8, 3)], [1, 3], [False, True], [False, True]):
        run_case(*shape, batch, transpose_a, transpose_b)
    for shape, transpose_a, transpose_b in itertools.product([(768, 128, 256), (128, 768, 256), (1024, 256, 128)], [False, True], [False, True]):
        run_case(*shape, 1, transpose_a, transpose_b)
    run_case(64, 64, 64, 3, False, False, repeats=30)
finally:
    destroy(handle)
    if stream.value:bind(hip,'hipStreamDestroy',[c.c_void_p])(stream)

supported = [result for result in results if result['supported']]
unsupported = [result for result in results if not result['supported']]
report = {'library': str(rocblas_path), 'sha256': hashlib.sha256(rocblas_path.read_bytes()).hexdigest(), 'implementation': 'Windows HIP small-GEMM workaround; Schaka-inspired, not the original Linux patch', 'cases': len(results), 'supported_cases': len(supported), 'unsupported_cases': len(unsupported), 'comparison_iterations': sum(result['repeats'] for result in supported), 'small_gemm_cases': 56, 'larger_gemm_cases': 12, 'reused_buffer_iterations': 30, 'max_abs_error': max(result['max_abs_error'] for result in supported), 'supported_results_match_cpu': True, 'full_shape_coverage': not unsupported, 'unsupported': unsupported, 'results': results}
report['api']=arguments.api
report['nondefault_stream']=arguments.nondefault_stream
output_path = Path(arguments.report).resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({key: value for key, value in report.items() if key != 'results'}), flush=True)

if unsupported and not arguments.allow_unsupported:
    raise SystemExit(2)
