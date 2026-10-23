# SPDX-License-Identifier: GPL-3.0-only
"""Actual loader checks: cold invalid handle, named/ordinal export agreement."""
import argparse,ctypes as c,importlib.util,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--hip-root',required=True);p.add_argument('--hip-runtime',required=True);p.add_argument('--dll',required=True);p.add_argument('--report',required=True);args=p.parse_args()
dll=Path(args.dll).resolve();bin=Path(args.hip_root)/'bin'
dirs=[os.add_dll_directory(str(x)) for x in [bin,dll.parent,Path(args.hip_runtime).parent]]
hip=c.WinDLL(args.hip_runtime);shim=c.WinDLL(str(dll))
fn=shim.rocblas_sgemm;fn.argtypes=[c.c_void_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_int,c.c_void_p,c.c_void_p,c.c_int,c.c_void_p,c.c_int,c.c_void_p,c.c_void_p,c.c_int];fn.restype=c.c_int
status=fn(None,111,111,4,4,4,None,None,4,None,4,None,None,4)
if status!=1:raise RuntimeError('Cold NULL handle must preserve invalid_handle, got '+str(status))
module_path=Path(__file__).resolve().parent.parent/'patches/small-gemm/make-exports.py'
spec=importlib.util.spec_from_file_location('exports',module_path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
exports=mod.exports(dll)
get=c.WinDLL('kernel32').GetProcAddress;get.argtypes=[c.c_void_p,c.c_void_p];get.restype=c.c_void_p
for name,(ordinal,forwarder) in exports.items():
 encoded=c.create_string_buffer(name.encode('ascii'))
 by_name=get(shim._handle,c.cast(encoded,c.c_void_p));by_ordinal=get(shim._handle,c.c_void_p(ordinal))
 if not by_name or by_name!=by_ordinal:raise RuntimeError('Loader export mismatch: '+name)
report={'passed':True,'exports_resolved':len(exports),'original_exports':3936,'cold_null_handle_status':status,'ordinals_match_names':True}
Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report))
