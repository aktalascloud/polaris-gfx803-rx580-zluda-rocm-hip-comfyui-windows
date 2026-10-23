# SPDX-License-Identifier: GPL-3.0-only
import ctypes as c
import json
from pathlib import Path

import argparse
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hip_runtime import resolve, describe
parser = argparse.ArgumentParser()
parser.add_argument('--zluda-bin', required=True)
parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
parser.add_argument('--hip-runtime')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--report', default='cuda-driver-api.json')
args = parser.parse_args()
runtime_path=resolve(args.root, args.hip_runtime)
root=Path(args.root).resolve()
handles=[os.add_dll_directory(str(x)) for x in [runtime_path.parent,root/'hip'/'bin',Path(args.zluda_bin).resolve()] if x.is_dir()]
handles.append(c.WinDLL(str(runtime_path)))
cuda = c.WinDLL(str(Path(args.zluda_bin).resolve() / 'nvcuda.dll'))


def function(name, arguments):
    result = getattr(cuda, name)
    result.argtypes = arguments
    result.restype = c.c_int
    return result


def check(result, operation):
    if result:
        raise RuntimeError(f'{operation}: CUDA error {result}')


initialize = function('cuInit', [c.c_uint])
get_device = function('cuDeviceGet', [c.POINTER(c.c_int), c.c_int])
get_name = function('cuDeviceGetName', [c.c_char_p, c.c_int, c.c_int])
create_context = function('cuCtxCreate_v2', [c.POINTER(c.c_void_p), c.c_uint, c.c_int])
destroy_context = function('cuCtxDestroy_v2', [c.c_void_p])
allocate = function('cuMemAlloc_v2', [c.POINTER(c.c_uint64), c.c_size_t])
release = function('cuMemFree_v2', [c.c_uint64])
copy_to_device = function('cuMemcpyHtoD_v2', [c.c_uint64, c.c_void_p, c.c_size_t])
copy_to_host = function('cuMemcpyDtoH_v2', [c.c_void_p, c.c_uint64, c.c_size_t])
load_module = function('cuModuleLoadData', [c.POINTER(c.c_void_p), c.c_void_p])
unload_module = function('cuModuleUnload', [c.c_void_p])
get_kernel = function('cuModuleGetFunction', [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p])
launch = function('cuLaunchKernel', [c.c_void_p, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_uint, c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p])
synchronize = function('cuCtxSynchronize', [])
check(initialize(0), 'cuInit')
device = c.c_int()
check(get_device(c.byref(device), args.device), 'cuDeviceGet')
name = c.create_string_buffer(256)
check(get_name(name, len(name), device), 'cuDeviceGetName')
context = c.c_void_p()
check(create_context(c.byref(context), 0, device), 'cuCtxCreate')
allocations = []
module = c.c_void_p()
try:
    count = 64
    array_type = c.c_float * count
    a, b, output = array_type(*range(count)), array_type(*(100 + index for index in range(count))), array_type()
    for host in [a, b, output]:
        address = c.c_uint64()
        check(allocate(c.byref(address), c.sizeof(host)), 'cuMemAlloc')
        allocations.append(address)
        check(copy_to_device(address, host, c.sizeof(host)), 'cuMemcpyHtoD')
    ptx = c.create_string_buffer(b'''
.version 6.4
.target sm_61
.address_size 64
.visible .entry vector_add(.param .u64 a, .param .u64 b, .param .u64 output, .param .u32 count)
{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<3>;
    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r4, [count];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r0, %r2, %r3, %r1;
    setp.ge.u32 %p, %r0, %r4;
    @%p bra done;
    mul.wide.u32 %rd4, %r0, 4;
    add.s64 %rd5, %rd1, %rd4;
    add.s64 %rd6, %rd2, %rd4;
    add.s64 %rd7, %rd3, %rd4;
    ld.global.f32 %f0, [%rd5];
    ld.global.f32 %f1, [%rd6];
    add.f32 %f2, %f0, %f1;
    st.global.f32 [%rd7], %f2;
done:
    ret;
}
''')
    check(load_module(c.byref(module), ptx), 'cuModuleLoadData PTX')
    kernel = c.c_void_p()
    check(get_kernel(c.byref(kernel), module, b'vector_add'), 'cuModuleGetFunction')
    length = c.c_uint(count)
    parameters = (c.c_void_p * 4)(*[c.addressof(value) for value in [*allocations, length]])
    check(launch(kernel, 1, 1, 1, count, 1, 1, 0, None, parameters, None), 'cuLaunchKernel')
    check(synchronize(), 'cuCtxSynchronize')
    check(copy_to_host(output, allocations[2], c.sizeof(output)), 'cuMemcpyDtoH')
    expected = [100 + 2 * index for index in range(count)]
    if list(output) != expected:
        raise RuntimeError('CUDA vector addition mismatch')
    get_module_path = c.WinDLL('kernel32').GetModuleFileNameW
    get_module_path.argtypes = [c.c_void_p, c.c_wchar_p, c.c_uint]
    path = c.create_unicode_buffer(1024)
    get_module_path(cuda._handle, path, len(path))
    report = {'device': name.value.decode(), 'nvcuda': path.value, 'api': 'CUDA Driver API', 'ptx_target': 'sm_61', 'elements': count, 'memory_copy_passed': True, 'ptx_compilation_passed': True, 'kernel_execution_passed': True, 'passed': True, 'hip_runtime': describe(runtime_path), 'models_used': False, 'torch_imported': 'torch' in sys.modules}
    destination = Path(args.report).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
finally:
    if module.value:
        unload_module(module)
    for address in allocations:
        release(address)
    destroy_context(context)
