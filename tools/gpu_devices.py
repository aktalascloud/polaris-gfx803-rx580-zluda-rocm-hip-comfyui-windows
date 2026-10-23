# SPDX-License-Identifier: GPL-3.0-only
"""DXGI and torch-directml use separate device-index namespaces."""
import ctypes as c
import json
import uuid


class Description(c.Structure):
    _fields_ = [('name', c.c_wchar * 128), ('vendor', c.c_uint), ('device', c.c_uint),
                ('subsystem', c.c_uint), ('revision', c.c_uint), ('video_memory', c.c_size_t),
                ('system_memory', c.c_size_t), ('shared_memory', c.c_size_t),
                ('luid_low', c.c_uint), ('luid_high', c.c_int), ('flags', c.c_uint)]


def method(pointer, index, result, *args):
    table = c.cast(pointer, c.POINTER(c.POINTER(c.c_void_p))).contents
    return c.WINFUNCTYPE(result, c.c_void_p, *args)(table[index])


def adapters():
    guid = (c.c_byte * 16).from_buffer_copy(uuid.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
    factory = c.c_void_p()
    create = c.WinDLL('dxgi').CreateDXGIFactory1
    create.argtypes = [c.c_void_p, c.POINTER(c.c_void_p)]
    create.restype = c.c_long
    if create(c.byref(guid), c.byref(factory)) < 0:
        raise RuntimeError('CreateDXGIFactory1 failed')
    rows = []
    try:
        enum = method(factory, 12, c.c_long, c.c_uint, c.POINTER(c.c_void_p))
        for index in range(64):
            adapter = c.c_void_p()
            status = enum(factory, index, c.byref(adapter))
            if status & 0xffffffff == 0x887a0002:
                break
            if status < 0:
                raise RuntimeError('EnumAdapters1 failed')
            try:
                desc = Description()
                if method(adapter, 10, c.c_long, c.POINTER(Description))(adapter, c.byref(desc)) < 0:
                    raise RuntimeError('GetDesc1 failed')
                rows.append(dict(dxgi_index=index, name=desc.name, vendor=hex(desc.vendor),
                                 pci_device=hex(desc.device), video_memory=desc.video_memory,
                                 luid=f'{desc.luid_high & 0xffffffff:08x}:{desc.luid_low:08x}',
                                 flags=desc.flags, software=bool(desc.flags & 2) or desc.vendor == 0x1414))
            finally:
                method(adapter, 2, c.c_ulong)(adapter)
    finally:
        method(factory, 2, c.c_ulong)(factory)
    return rows


def radeon(index=None, rows=None):
    rows = adapters() if rows is None else rows
    candidates = [r for r in rows if r['vendor'] == '0x1002' and not r['software']]
    if index is not None and index >= 0:
        candidates = [r for r in candidates if r['dxgi_index'] == index]
    if len(candidates) != 1:
        raise RuntimeError('Select one physical AMD adapter with --index / -DxgiDevice. Candidates: ' + json.dumps(candidates))
    return candidates[0]


def torch_radeon(dml, selected, index=None):
    def key(name):
        return ''.join(name.rstrip('\x00').lower().split())
    devices = [dict(torch_index=i, name=dml.device_name(i).rstrip('\x00')) for i in range(dml.device_count())]
    matches = [r for r in devices if key(r['name']) == key(selected['name'])]
    if index is not None:
        matches = [r for r in matches if r['torch_index'] == index]
    if len(matches) != 1:
        raise RuntimeError('Choose a torch-directml index separately; DXGI indices cannot be reused: ' + json.dumps(devices))
    return matches[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=-1)
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    print(json.dumps(radeon(args.index) if args.select else adapters()))
