# SPDX-License-Identifier: GPL-3.0-only
"""Isolated rocBLAS correctness/steady-state GEMM probe. Explicit manual use only."""
import argparse,ctypes,json,os,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hip_runtime import resolve
p=argparse.ArgumentParser()
p.add_argument('--hip',required=True);p.add_argument('--zluda',required=True)
p.add_argument('--dll',required=True);p.add_argument('--kernels',required=True)
p.add_argument('--label',required=True);p.add_argument('--cache',required=True)
p.add_argument('--hip-runtime')
p.add_argument('--suite',choices=['general','diffusion'],default='general')
a=p.parse_args()
hip,zluda,dll,kernels=[Path(v).resolve() for v in [a.hip,a.zluda,a.dll,a.kernels]]
for path in [hip/'bin',zluda,dll.parent,kernels]:
 if not path.exists():raise RuntimeError('Missing component folder')
os.environ.update(HIP_PATH=str(hip),ROCBLAS_TENSILE_LIBPATH=str(kernels),ZLUDA_CACHE_DIR=a.cache,DISABLE_ADDMM_CUDA_LT='1',CUDA_MODULE_LOADING='LAZY',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
for key in ['HSA_OVERRIDE_GFX_VERSION','GPU_FORCE_BLIT_COPY_SIZE']:os.environ.pop(key,None)
handles=[os.add_dll_directory(str(x)) for x in [dll.parent,hip/'bin',zluda]]
runtime=resolve(hip.parent,a.hip_runtime)
handles.append(os.add_dll_directory(str(runtime.parent)))
handles.extend([ctypes.WinDLL(str(runtime)),ctypes.WinDLL(str(dll)),ctypes.WinDLL(str(zluda/'nvcuda.dll')),ctypes.WinDLL(str(zluda/'cublas.dll'))])
def emit(**v):print(json.dumps({'profile':a.label,**v}),flush=True)
emit(stage='libraries_loaded')
import torch
torch.set_num_threads(8);torch.set_num_interop_threads(1)
torch.backends.cudnn.enabled=False
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if hasattr(torch.backends.cuda,'preferred_blas_library'):torch.backends.cuda.preferred_blas_library('cublas')
if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
free,total=torch.cuda.mem_get_info()
emit(stage='device',torch=torch.__version__,device=torch.cuda.get_device_name(0),free_vram=free,total_vram=total)
if free<768*1024**2:raise RuntimeError('Insufficient free VRAM; no existing application will be stopped')
generator=torch.Generator().manual_seed(580)
dtypes=[torch.float32] if a.suite=='diffusion' else [torch.float32,torch.float16]
shapes=[(320,4096,320),(1280,1024,1280),(320,4096,2880),(768,77,768)] if a.suite=='diffusion' else [(127,65,96),(2560,21,2560),(9728,21,2560),(512,512,512)]
for dtype in dtypes:
 for m,n,k in shapes:
  cpu_a=(torch.rand((m,k),generator=generator)-0.5).to(dtype)
  cpu_b=(torch.rand((k,n),generator=generator)-0.5).to(dtype)
  reference=(cpu_a.float()@cpu_b.float()).to(dtype)
  ga,gb=cpu_a.cuda(),cpu_b.cuda()
  result=ga@gb;torch.cuda.synchronize()
  out=result.cpu();err=(out.float()-reference.float()).abs()
  atol,rtol=(5e-5,2e-4) if dtype==torch.float32 else (0.0625,0.02)
  ok=bool(torch.allclose(out,reference,atol=atol,rtol=rtol))
  emit(stage='numeric',dtype=str(dtype),shape=[m,n,k],max_abs_error=err.max().item(),rms_error=err.square().mean().sqrt().item(),atol=atol,rtol=rtol,status='PASS' if ok else 'FAIL')
  if not ok:raise SystemExit(2)
  # Five warmups, seven samples of five GEMMs; transfers and initial JIT excluded.
  for _ in range(5):result=ga@gb
  torch.cuda.synchronize();samples=[]
  for _ in range(7):
   start=time.perf_counter()
   for __ in range(5):result=ga@gb
   torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000/5)
  emit(stage='timing',dtype=str(dtype),shape=[m,n,k],median_ms=statistics.median(samples),samples_ms=samples)
  del ga,gb,result,cpu_a,cpu_b,reference,out,err
emit(stage='complete',status='PASS')
