# SPDX-License-Identifier: GPL-3.0-only
"""Resolve an existing x64 HIP runtime, without installing or copying it."""
import ctypes
import hashlib
import json
import os
import struct
from pathlib import Path


def validate(path):
    path = Path(path).resolve(strict=True)
    if path.name.lower() != 'amdhip64.dll':
        raise ValueError('Select an existing amdhip64.dll, not another library')
    with path.open('rb') as stream:
        if stream.read(2) != b'MZ':
            raise ValueError('HIP runtime is not a PE image')
        stream.seek(0x3c)
        offset = struct.unpack('<I', stream.read(4))[0]
        stream.seek(offset)
        if stream.read(6) != b'PE\0\0\x64\x86':
            raise ValueError('HIP runtime must be a Windows x64 PE image')
    return path


def resolve(root, explicit=None):
    root = Path(root)
    choice = explicit or os.environ.get('AI_HIP_RUNTIME')
    if not choice and (root / 'profile.json').exists():
        choice = json.loads((root / 'profile.json').read_text(encoding='utf-8-sig')).get('hip_runtime')
    if choice:
        return validate(choice)  # Explicit errors never silently fall back.
    system = Path(os.environ['SystemRoot']) / 'System32' / 'amdhip64.dll'
    if system.exists():
        return validate(system)
    return validate(root / 'hip' / 'bin' / 'amdhip64.dll')


def version(path):
    lib = ctypes.WinDLL('version', use_last_error=True)
    lib.GetFileVersionInfoSizeW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
    size = lib.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    data = ctypes.create_string_buffer(size)
    lib.GetFileVersionInfoW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    if not lib.GetFileVersionInfoW(str(path), 0, size, data):
        return None
    pointer = ctypes.c_void_p()
    length = ctypes.c_uint()
    lib.VerQueryValueW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
    if not lib.VerQueryValueW(data, '\\', ctypes.byref(pointer), ctypes.byref(length)) or length.value < 16:
        return None
    values = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_uint32))
    ms, ls = values[2], values[3]
    return '.'.join(str(x) for x in (ms >> 16, ms & 65535, ls >> 16, ls & 65535))


def describe(path):
    path = validate(path)
    return dict(path=str(path), architecture='x64', version=version(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument('--runtime')
    args = parser.parse_args()
    print(json.dumps(describe(resolve(args.root, args.runtime))))
