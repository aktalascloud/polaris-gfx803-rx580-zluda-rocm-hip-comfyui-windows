# SPDX-License-Identifier: GPL-3.0-only
"""Compare guarded edge modes with the original backend; no model files."""
import argparse,ctypes as c,json,os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--hip-root',required=True);p.add_argument('--hip-runtime',required=True);p.add_argument('--dll',required=True);p.add_argument('--report',required=True);args=p.parse_args()
bin=Path(args.hip_root)/'bin';dll=Path(args.dll).resolve()
os.environ['ROCBLAS_TENSILE_LIBPATH']=str(bin/'rocblas/library')
handles=[os.add_dll_directory(str(x)) for x in [bin,dll.parent,Path(args.hip_runtime).parent]]
hip=c.WinDLL(args.hip_runtime);backend=c.WinDLL(str(dll.parent/'rocblas_backend.dll'));shim=c.WinDLL(str(dll))
def bind(lib,name,types):
 fn=getattr(lib,name);fn.argtypes=types;fn.restype=c.c_int;return fn
P=c.c_void_p;I=c.c_int;L=c.c_int64;F=c.c_float;PF=c.POINTER(F)
create=bind(backend,'rocblas_create_handle',[c.POINTER(P)]);destroy=bind(backend,'rocblas_destroy_handle',[P]);h=P()
def checked(code):
 if code:raise RuntimeError('API status '+str(code))
checked(create(c.byref(h)))
alloc=bind(hip,'hipMalloc',[c.POINTER(P),c.c_size_t]);free=bind(hip,'hipFree',[P]);copy=bind(hip,'hipMemcpy',[P,P,c.c_size_t,I]);sync=bind(hip,'hipDeviceSynchronize',[])
sgtypes=[P,I,I,I,I,I,PF,P,I,P,I,PF,P,I]
sttypes=[P,I,I,I,I,I,PF,P,I,L,P,I,L,PF,P,I,L,I]
extypes=[P,I,I,I,I,I,P,P,I,I,P,I,I,P,P,I,I,P,I,I,I,I,I,c.c_uint]
stextypes=[P,I,I,I,I,I,P,P,I,I,L,P,I,I,L,P,P,I,I,L,P,I,I,L,I,I,I,I,c.c_uint]
def gemm(lib,api,m,n,k,batch,alpha,beta,a,b,cc,d,lda,ldb,ldc,ldd,sa,sb,sc,sd,ta=111,tb=111,dtype=151,compute=151):
 if api=='sgemm':return bind(lib,'rocblas_sgemm',sgtypes)(h,ta,tb,m,n,k,alpha,a,lda,b,ldb,beta,cc,ldc)
 if api=='strided':return bind(lib,'rocblas_sgemm_strided_batched',sttypes)(h,ta,tb,m,n,k,alpha,a,lda,sa,b,ldb,sb,beta,cc,ldc,sc,batch)
 if api=='ex':return bind(lib,'rocblas_gemm_ex',extypes)(h,ta,tb,m,n,k,alpha,a,dtype,lda,b,dtype,ldb,beta,cc,dtype,ldc,d,dtype,ldd,compute,0,0,0)
 return bind(lib,'rocblas_gemm_strided_batched_ex',stextypes)(h,ta,tb,m,n,k,alpha,a,dtype,lda,sa,b,dtype,ldb,sb,beta,cc,dtype,ldc,sc,d,dtype,ldd,sd,batch,compute,0,0,0)
rows=[];rng=np.random.default_rng(580)
try:
 for api in ['sgemm','strided','ex','strided-ex']:
  for av,bv in [(0.,0.),(0.,1.),(0.,.25),(1.,0.),(.75,1.),(-1.,.5)]:
   m,n,k=3,2,5;batch=3 if api in ['strided','strided-ex'] else 1
   lda,ldb,ldc,ldd=5,8,4,7;sa,sb,sc,sd=0,0,ldc*n+3,ldd*n+5
   aa=np.zeros(k*lda,dtype=np.float32);bb=np.zeros(n*ldb,dtype=np.float32)
   a=rng.uniform(-.5,.5,(m,k)).astype(np.float32);b=rng.uniform(-.5,.5,(k,n)).astype(np.float32)
   for j in range(k):aa[j*lda:j*lda+m]=a[:,j]
   for j in range(n):bb[j*ldb:j*ldb+k]=b[:,j]
   cc=np.full(batch*sc,12345.,dtype=np.float32);dd=np.full(batch*sd,12345.,dtype=np.float32);initial=[]
   for z in range(batch):
    cx=rng.uniform(-.5,.5,(m,n)).astype(np.float32);initial.append(cx)
    for j in range(n):cc[z*sc+j*ldc:z*sc+j*ldc+m]=cx[:,j]
   allocations=[]
   try:
    for buf in [aa,bb,cc,dd]:
     ptr=P();checked(alloc(c.byref(ptr),buf.nbytes));allocations.append(ptr);checked(copy(ptr,buf.ctypes.data,buf.nbytes,1))
    alpha,beta=F(av),F(bv)
    # Null A/B with alpha=0 must not be read; padding remains untouched.
    ap,bp=allocations[:2] if av else [None,None]
    cp=allocations[2];dp=allocations[3] if api in ['ex','strided-ex'] else cp
    checked(gemm(shim,api,m,n,k,batch,c.byref(alpha),c.byref(beta),ap,bp,cp,dp,lda,ldb,ldc,ldd,sa,sb,sc,sd));checked(sync())
    host=dd.copy() if api in ['ex','strided-ex'] else cc.copy();checked(copy(host.ctypes.data,dp,host.nbytes,2))
    out_ld,out_stride=(ldd,sd) if api in ['ex','strided-ex'] else (ldc,sc)
    max_error=0.;mask=np.zeros(host.size,dtype=bool)
    for z in range(batch):
     actual=np.column_stack([host[z*out_stride+j*out_ld:z*out_stride+j*out_ld+m] for j in range(n)])
     expected=(av*(a.astype(np.float64)@b.astype(np.float64))+bv*initial[z]).astype(np.float32)
     np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=3e-5);max_error=max(max_error,float(np.max(abs(actual-expected))))
     for j in range(n):mask[z*out_stride+j*out_ld:z*out_stride+j*out_ld+m]=True
    if not np.all(host[~mask]==12345.):raise RuntimeError('Padding or batch gap overwritten')
    rows.append(dict(api=api,alpha=av,beta=bv,broadcast_inputs=True,padded=True,max_abs_error=max_error,passed=True))
   finally:
    for ptr in allocations:checked(free(ptr))
 # Validate modes and status forwarding, using small valid NN matrices.
 buffers=[]
 for _ in range(3):
  ptr=P();checked(alloc(c.byref(ptr),64));buffers.append(ptr)
 host=np.arange(16,dtype=np.float32)/17
 for ptr in buffers:checked(copy(ptr,host.ctypes.data,64,1))
 alpha,beta=F(.75),F(.25)
 def simple(lib,m=4,ta=111,alpha_ptr=None):
  return gemm(lib,'sgemm',m,4,4,1,c.byref(alpha) if alpha_ptr is None else alpha_ptr,c.byref(beta),*buffers,buffers[2],4,4,4,4,0,0,0,0,ta=ta)
 for name,kwargs in [('zero_m',{'m':0}),('negative_m',{'m':-1}),('invalid_transpose',{'ta':-1}),('null_alpha',{'alpha_ptr':PF()})]:
  first=simple(backend,**kwargs);second=simple(shim,**kwargs)
  if first!=second:raise RuntimeError('Status mismatch: '+name)
  rows.append(dict(mode=name,backend_status=first,shim_status=second,passed=True))
 # Device scalar mode: the shim must not dereference either GPU scalar on the host.
 scalar=[]
 for value in [alpha,beta]:
  ptr=P();checked(alloc(c.byref(ptr),4));scalar.append(ptr);checked(copy(ptr,c.byref(value),4,1))
 checked(bind(backend,'rocblas_set_pointer_mode',[P,I])(h,1))
 first=gemm(backend,'sgemm',4,4,4,1,c.cast(scalar[0],PF),c.cast(scalar[1],PF),*buffers,buffers[2],4,4,4,4,0,0,0,0);checked(sync())
 for ptr in buffers:checked(copy(ptr,host.ctypes.data,64,1))
 second=gemm(shim,'sgemm',4,4,4,1,c.cast(scalar[0],PF),c.cast(scalar[1],PF),*buffers,buffers[2],4,4,4,4,0,0,0,0);checked(sync())
 if first!=second:raise RuntimeError('Device scalar status changed')
 output=np.empty(16,np.float32);checked(copy(output.ctypes.data,buffers[2],64,2))
 matrix=host.reshape(4,4).T;np.testing.assert_allclose(output.reshape(4,4).T,.75*(matrix@matrix)+.25*matrix,rtol=1e-4,atol=3e-5)
 rows.append(dict(mode='device_scalars',backend_status=first,shim_status=second,max_abs_error=float(np.max(abs(output.reshape(4,4).T-(.75*(matrix@matrix)+.25*matrix)))),passed=True))
 checked(bind(backend,'rocblas_set_pointer_mode',[P,I])(h,0))
 for ptr in scalar:checked(free(ptr))
 for lib,label in [(backend,'backend'),(shim,'shim')]:
  checked(bind(backend,'rocblas_start_device_memory_size_query',[P])(h));status=simple(lib);size=c.c_size_t()
  checked(bind(backend,'rocblas_stop_device_memory_size_query',[P,c.POINTER(c.c_size_t)])(h,c.byref(size)))
  rows.append(dict(mode='memory_size_query',library=label,status=status,bytes=size.value))
 queries=[r for r in rows if r.get('mode')=='memory_size_query']
 if queries[0]['status']!=queries[1]['status'] or queries[0]['bytes']!=queries[1]['bytes']:raise RuntimeError('Size-query forwarding changed')
 # Capture only; no graph is launched. Compare original and shim status.
 stream=P();checked(bind(hip,'hipStreamCreateWithFlags',[c.POINTER(P),c.c_uint])(c.byref(stream),1));checked(bind(backend,'rocblas_set_stream',[P,P])(h,stream))
 captures=[]
 for lib,label in [(backend,'backend'),(shim,'shim')]:
  begin=bind(hip,'hipStreamBeginCapture',[P,I])(stream,0)
  if begin:raise RuntimeError('Capture not available: '+str(begin))
  status=simple(lib);graph=P();end=bind(hip,'hipStreamEndCapture',[P,c.POINTER(P)])(stream,c.byref(graph))
  if graph:checked(bind(hip,'hipGraphDestroy',[P])(graph))
  captures.append(dict(library=label,status=status,end_status=end))
 if captures[0]['status']!=captures[1]['status'] or captures[0]['end_status']!=captures[1]['end_status']:raise RuntimeError('Capture forwarding changed')
 rows.append(dict(mode='graph_capture',results=captures,passed=True))
 checked(bind(backend,'rocblas_set_stream',[P,P])(h,None));checked(bind(hip,'hipStreamDestroy',[P])(stream))
 for ptr in buffers:checked(free(ptr))
finally:checked(destroy(h))
report={'checks':rows,'passed':True,'scope':'Scalar/padding/broadcast/mode contracts; no performance or multi-context claim'}
Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
