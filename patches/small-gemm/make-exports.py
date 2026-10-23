# SPDX-License-Identifier: GPL-3.0-only
"""Generate and independently verify exact named exports and original ordinals."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
EXPECTED='9e88918e65b81eeb128c69cd7a96ab495b101daf51f6f78f943445d7d4c5f70d'
OVERRIDES={'rocblas_sgemm','rocblas_sgemm_strided_batched','rocblas_gemm_ex','rocblas_gemm_strided_batched_ex'}
def exports(path, desired=None):
 data=bytearray(Path(path).read_bytes())
 if data[:2]!=b'MZ':raise ValueError('Not a PE image')
 pe=struct.unpack_from('<I',data,0x3c)[0]
 if data[pe:pe+6]!=b'PE\0\0\x64\x86':raise ValueError('Expected Windows x64 PE')
 sections=struct.unpack_from('<H',data,pe+6)[0];optional=pe+24;size=struct.unpack_from('<H',data,pe+20)[0]
 if struct.unpack_from('<H',data,optional)[0]!=0x20b:raise ValueError('Expected PE32+')
 table=[]
 for i in range(sections):
  virtual,address,raw_size,raw=struct.unpack_from('<IIII',data,optional+size+40*i+8)
  table.append((address,max(virtual,raw_size),raw))
 def offset(address):
  for start,length,raw in table:
   if start<=address<start+length:
    pos=raw+address-start
    if pos>=len(data):raise ValueError('RVA outside file')
    return pos
  raise ValueError('Invalid RVA')
 def string(address):
  start=offset(address);return data[start:data.index(b'\0',start)].decode('ascii')
 er,es=struct.unpack_from('<II',data,optional+112)
 _,_,_,_,_,base,fc,nc,functions,names,ordinals=struct.unpack_from('<IIHHIIIIIII',data,offset(er))
 result={};addresses={};positions={};used=set()
 for i in range(nc):
  name=string(struct.unpack_from('<I',data,offset(names)+i*4)[0]);index=struct.unpack_from('<H',data,offset(ordinals)+i*2)[0]
  addr=struct.unpack_from('<I',data,offset(functions)+index*4)[0]
  result[name]=(base+index,string(addr) if er<=addr<er+es else None)
  addresses[name]=addr;positions[name]=offset(ordinals)+i*2;used.add(index)
 if any(struct.unpack_from('<I',data,offset(functions)+i*4)[0] for i in range(fc) if i not in used):raise ValueError('Unaccounted ordinal-only exports')
 if desired is not None:
  if set(desired)!=set(result):raise ValueError('Unexpected names before ordinal repair')
  new_base=min(desired.values());new_count=max(desired.values())-new_base+1
  if new_count>fc or new_count>65535:raise ValueError('Insufficient existing export storage')
  # AMD LLD 17 assigns fresh ordinals to .def forwarders. Reindex its existing
  # EAT and name-ordinal table in place, without moving names/forwarder strings.
  for i in range(fc):struct.pack_into('<I',data,offset(functions)+i*4,0)
  for name,ordinal in desired.items():
   struct.pack_into('<I',data,offset(functions)+(ordinal-new_base)*4,addresses[name])
   struct.pack_into('<H',data,positions[name],ordinal-new_base)
  struct.pack_into('<II',data,offset(er)+16,new_base,new_count)
  Path(path).write_bytes(data)
  return exports(path)
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--backend',required=True);parser.add_argument('--output');parser.add_argument('--verify-shim');parser.add_argument('--repair-shim');args=parser.parse_args()
 original=Path(args.backend)
 if hashlib.sha256(original.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Supply the original known advanced-lvl-up backend, never a shim')
 old=exports(original)
 if len(old)!=3936 or not OVERRIDES.issubset(old):raise ValueError('Unexpected backend exports')
 if args.repair_shim:
  desired={name:value[0] for name,value in old.items()};desired['rocblas_windows_small_gemm_patch_version']=max(desired.values())+1
  exports(args.repair_shim,desired)
  args.verify_shim=args.repair_shim
 if args.verify_shim:
  new=exports(args.verify_shim)
  if set(new)!=set(old)|{'rocblas_windows_small_gemm_patch_version'}:raise ValueError('Export names changed')
  for name,(ordinal,_) in old.items():
   actual,forward=new[name]
   expected=None if name in OVERRIDES else 'rocblas_backend.'+name
   if ordinal!=actual or forward!=expected:raise ValueError('Export/ordinal mismatch: '+name)
  print(json.dumps({'verified_original_exports':len(old),'forwarded':3932,'overridden':4,'ordinals_preserved':True}));return
 if not args.output:parser.error('--output or --verify-shim is required')
 lines=['LIBRARY rocblas.dll','EXPORTS']
 for name,(ordinal,_) in old.items():lines.append('  '+name+('' if name in OVERRIDES else '=rocblas_backend.'+name)+' @'+str(ordinal))
 lines.append('  rocblas_windows_small_gemm_patch_version @'+str(max(v[0] for v in old.values())+1))
 Path(args.output).write_text('\n'.join(lines)+'\n',encoding='ascii')
 print('FORWARDED_EXPORTS 3932 PATCHED_EXPORTS 4')
if __name__=='__main__':main()
