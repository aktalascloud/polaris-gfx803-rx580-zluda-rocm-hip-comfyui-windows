# SPDX-License-Identifier: GPL-3.0-only
"""No GPU initialization; exercise adapter namespaces and runtime rejection."""
import json
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tools'))
from gpu_devices import radeon, torch_radeon
from hip_runtime import resolve, validate
from gpu_runtime import verify_rocblas


class SelectionTests(unittest.TestCase):
    rows=[dict(dxgi_index=0,name='Software',vendor='0x1414',software=True),
          dict(dxgi_index=1,name='AMD Radeon RX 580',vendor='0x1002',software=False)]

    def test_software_zero_is_not_selected(self):
        self.assertEqual(radeon(rows=self.rows)['dxgi_index'],1)
        with self.assertRaises(RuntimeError):radeon(0,rows=self.rows)

    def test_distinct_torch_namespace(self):
        class Dml:
            @staticmethod
            def device_count():return 1
            @staticmethod
            def device_name(index):return 'AMD Radeon RX 580\x00'
        self.assertEqual(torch_radeon(Dml,radeon(rows=self.rows))['torch_index'],0)
        with self.assertRaises(RuntimeError):torch_radeon(Dml,radeon(rows=self.rows),1)

    def test_multiple_physical_adapters_need_selection(self):
        rows=self.rows+[dict(dxgi_index=2,name='AMD Radeon RX 570',vendor='0x1002',software=False)]
        with self.assertRaises(RuntimeError):radeon(rows=rows)
        self.assertEqual(radeon(2,rows=rows)['dxgi_index'],2)

    def test_explicit_missing_runtime_never_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):resolve(folder,Path(folder)/'missing'/'amdhip64.dll')

    def test_runtime_architecture_and_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'amdhip64.dll'
            data=bytearray(128);data[:2]=b'MZ';struct.pack_into('<I',data,60,64);data[64:70]=b'PE\0\0\x4c\x01';path.write_bytes(data)
            with self.assertRaises(ValueError):validate(path)
            data[68:70]=b'\x64\x86';path.write_bytes(data)
            self.assertEqual(validate(path),path)
            renamed=path.with_name('other.dll');renamed.write_bytes(data)
            with self.assertRaises(ValueError):validate(renamed)

    def test_profile_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'profile.json').write_text(json.dumps({'hip_runtime':str(root/'profile'/'amdhip64.dll')}))
            with patch.dict(os.environ,{'AI_HIP_RUNTIME':str(root/'environment'/'amdhip64.dll')}):
                with self.assertRaises(FileNotFoundError) as raised:resolve(root)
                self.assertIn('environment',str(raised.exception))

    def test_rocblas_rejects_replaced_kernel_with_matching_dll(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            library=root/'rocblas'/'library'
            library.mkdir(parents=True)
            (root/'rocblas.dll').write_bytes(b'baseline')
            kernel=library/'gfx803.hsaco'
            kernel.write_bytes(b'matched')
            dll_hash=hashlib.sha256(b'baseline').hexdigest()
            library_hash=hashlib.sha256(b'gfx803.hsaco\0'+hashlib.sha256(b'matched').digest()).hexdigest()
            with patch('gpu_runtime.ROCBLAS_DLL_SHA256',dll_hash), patch('gpu_runtime.ROCBLAS_LIBRARY_SHA256',library_hash):
                self.assertFalse(verify_rocblas(root))
                kernel.write_bytes(b'mismatched')
                with self.assertRaisesRegex(ValueError, 'library does not match'):
                    verify_rocblas(root)

    def test_rocblas_variant_verifies_shared_library_and_shim(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            library=root/'shared'
            library.mkdir()
            (library/'gfx803.hsaco').write_bytes(b'matched')
            files={'rocblas.dll':b'shim','rocblas_backend.dll':b'baseline',
                   'rocblas_small_gemm.hsaco':b'kernel'}
            hashes={}
            for name, data in files.items():
                (root/name).write_bytes(data)
                hashes[name]=hashlib.sha256(data).hexdigest()
            (root/'small-gemm-patch.json').write_text(json.dumps({'enabled':True,'files':hashes}))
            library_hash=hashlib.sha256(b'gfx803.hsaco\0'+hashlib.sha256(b'matched').digest()).hexdigest()
            with patch('gpu_runtime.ROCBLAS_DLL_SHA256',hashes['rocblas_backend.dll']), patch('gpu_runtime.ROCBLAS_LIBRARY_SHA256',library_hash):
                self.assertTrue(verify_rocblas(root,library=library))
                (root/'rocblas_small_gemm.hsaco').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Small-GEMM artifact changed'):
                    verify_rocblas(root,library=library)


if __name__=='__main__':unittest.main()
