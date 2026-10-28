# SPDX-License-Identifier: GPL-3.0-only
"""Synthetic attention checks and optional comparison with pinned ComfyUI math."""
import argparse,ast,ctypes,hashlib,importlib.util,json,os,statistics,sys,time,types
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
parser.add_argument('--hip-root');parser.add_argument('--zluda-bin');parser.add_argument('--hip-runtime')
parser.add_argument('--implementation',choices=['tiled','default'],default='tiled')
parser.add_argument('--comfy-root');parser.add_argument('--benchmark',action='store_true')
parser.add_argument('--report',required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root/'tools'))
report_path=Path(args.report).resolve();report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_text(json.dumps({'passed':False,'status':'running','models_used':False}),encoding='utf-8')
handles=[]
if args.device=='cuda' or args.hip_root:
 from hip_runtime import resolve
 hip=Path(args.hip_root).resolve();zluda=Path(args.zluda_bin).resolve();runtime=resolve(hip.parent,args.hip_runtime)
 os.environ['ZLUDA_COMGR_LOG_LEVEL']='1'
 os.environ.update(HIP_PATH=str(hip),ROCBLAS_TENSILE_LIBPATH=str(hip/'bin/rocblas/library'),ZLUDA_CACHE_DIR=str(report_path.parent/'cache'),DISABLE_ADDMM_CUDA_LT='1',CUDA_MODULE_LOADING='LAZY')
 for key in ['HSA_OVERRIDE_GFX_VERSION','GPU_FORCE_BLIT_COPY_SIZE']:os.environ.pop(key,None)
 handles=[os.add_dll_directory(str(p)) for p in [runtime.parent,hip/'bin',zluda]]
 handles.extend([ctypes.WinDLL(str(runtime)),ctypes.WinDLL(str(hip/'bin/rocblas.dll')),ctypes.WinDLL(str(zluda/'nvcuda.dll')),ctypes.WinDLL(str(zluda/'cublas.dll'))])
import torch
from polaris_attention import attention
sys.path.insert(0,str(Path(__file__).resolve().parent))
from attention_baseline import load as load_baseline, dispatch
if args.implementation=='default':
 if args.device!='cuda':raise ValueError('Default dispatch validation targets CUDA; standalone CPU math uses --implementation tiled')
 context,_=load_baseline(args.comfy_root);attention=dispatch(context)['attention_sub_quad']
torch.set_num_threads(4)
torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if args.device=='cuda':
 assert torch.cuda.is_available();device_name=torch.cuda.get_device_name(0)
 assert '580' in device_name and 'ZLUDA' in device_name
else:device_name='CPU'

def reference(q,k,v,mask):
 scores=q.double()@k.double().transpose(-1,-2)/(q.shape[-1]**.5)
 if mask is not None:
  if mask.ndim==1:mask=mask.reshape(1,1,1,-1)
  elif mask.ndim==2:mask=mask.reshape(1,1,*mask.shape)
  elif mask.ndim==3:mask=mask.unsqueeze(1)
  if mask.dtype==torch.bool:scores=scores.masked_fill(~mask,-torch.inf)
  else:scores=scores+mask.double()
 blocked=torch.isneginf(scores).all(-1,keepdim=True)
 scores=scores.masked_fill(blocked,0)
 probabilities=torch.softmax(scores,dim=-1).masked_fill(blocked,0)
 return probabilities@v.double()

print(json.dumps({'stage':'initialized','device':device_name}),flush=True)
g=torch.Generator().manual_seed(803);records=[]
dtypes=[torch.float32,torch.float16] if args.device=='cuda' else [torch.float32]
with torch.inference_mode():
 for dtype in dtypes:
  for packed in [False,True]:
   for separate in [False,True]:
    for upcast in ([False,True] if dtype==torch.float16 else [True]):
     for mask_kind in ['none','bool-key','bool-qk','bool-batch','bool-full','bias-qk','bias-full','bias-key','bool-head','all-blocked','single-key']:
      print(json.dumps({'stage':'case','index':len(records)+1,'dtype':str(dtype),'packed':packed,'output4d':separate,'upcast':upcast,'mask':mask_kind}),flush=True)
      b,h,t,s,d,dv=2,3,13,17,16,16
      q=torch.randn((b,h,t,d),generator=g).to(dtype);k=torch.randn((b,h,s,d),generator=g).to(dtype);v=torch.randn((b,h,s,dv),generator=g).to(dtype)
      mask=None
      if mask_kind=='bool-key':mask=torch.rand((s,),generator=g)>.35
      elif mask_kind=='bool-qk':mask=torch.rand((t,s),generator=g)>.35
      elif mask_kind=='bool-batch':mask=torch.rand((b,t,s),generator=g)>.35
      elif mask_kind=='bool-full':mask=torch.rand((b,h,t,s),generator=g)>.35
      elif mask_kind=='bias-qk':mask=torch.randn((t,s),generator=g)*.2;mask[0]=-torch.inf
      elif mask_kind=='bias-full':mask=torch.randn((b,h,t,s),generator=g)*.2;mask[:,:,0]=-torch.inf
      elif mask_kind=='bias-key':mask=torch.randn((b,1,1,s),generator=g)*.2
      elif mask_kind=='bool-head':mask=torch.rand((1,h,1,s),generator=g)>.4
      elif mask_kind=='all-blocked':mask=torch.zeros((b,1,t,s),dtype=torch.bool)
      elif mask_kind=='single-key':mask=torch.zeros((b,1,1,s),dtype=torch.bool);mask[0,:,:,0]=True;mask[1,:,:,-1]=True
      expected=reference(q,k,v,mask)
      def layout(x):return x if packed else x.transpose(1,2).reshape(b,x.shape[2],-1)
      inputs=[layout(x).to(args.device) for x in [q,k,v]]
      # Force non-contiguous input strides without changing values.
      inputs=[x.transpose(-1,-2).contiguous().transpose(-1,-2) for x in inputs]
      actual=attention(*inputs,h,mask=None if mask is None else mask.to(args.device),attn_precision=torch.float32 if upcast else None,skip_reshape=packed,skip_output_reshape=separate,**(dict(query_chunk_size=5,head_chunk_size=2,score_limit_bytes=4096) if args.implementation=='tiled' else {})).cpu()
      expected=expected if separate else expected.transpose(1,2).reshape(b,t,-1)
      atol,rtol=(3e-5,1e-4) if dtype==torch.float32 else (3e-3,2e-3)
      torch.testing.assert_close(actual.double(),expected,atol=atol,rtol=rtol)
      assert actual.dtype==dtype and actual.shape==expected.shape
      if mask_kind=='all-blocked':assert torch.count_nonzero(actual)==0
      if mask_kind=='single-key':
       expected_single=torch.stack([v[0,:,0],v[1,:,-1]])[:,:,None,:].expand(b,h,t,dv)
       expected_single=expected_single if separate else expected_single.transpose(1,2).reshape(b,t,-1)
       torch.testing.assert_close(actual,expected_single,atol=0,rtol=0)
      records.append(dict(dtype=str(dtype),input_4d=packed,output_4d=separate,requested_upcast=upcast,effective_compute_dtype='torch.float32',mask=mask_kind,noncontiguous=True,max_abs_error=float((actual.double()-expected).abs().max()),atol=atol,rtol=rtol,passed=True))
 # Tiny masked shapes must not rely on a missing Tensile solution.
 for t,s,d in [(1,1,1),(2,2,1),(7,3,5)]:
  q=torch.randn((1,3,t,d),generator=g);k=torch.randn((1,3,s,d),generator=g);v=torch.randn((1,3,s,4),generator=g)
  mask=torch.ones((t,s),dtype=torch.bool)
  out=attention(q.to(args.device),k.to(args.device),v.to(args.device),3,mask=mask.to(args.device),skip_reshape=True,skip_output_reshape=True).cpu()
  torch.testing.assert_close(out.double(),reference(q,k,v,mask),atol=3e-5,rtol=1e-4)
  records.append(dict(edge='tiny-masked',shape=[t,s,d],passed=True))
 if args.implementation=='default':
  q=torch.randn((1,2,9,16),generator=g);k=torch.randn((1,2,11,16),generator=g);v=torch.randn((1,2,11,16),generator=g)
  out=attention(query=q.cuda(),key=k.cuda(),value=v.cuda(),heads=2,skip_reshape=True,skip_output_reshape=True).cpu()
  torch.testing.assert_close(out.double(),reference(q,k,v,None),atol=3e-5,rtol=1e-4)
  records.append(dict(edge='upstream-keyword-names',passed=True))
 # Batch/head/query tails exercise the default tiles on realistic key counts.
 for dtype in dtypes:
  for kind in ['none','bool','bias']:
   b,h,t,s,d=2,11,259,521,16
   qs=torch.randn((b,t,h*d),generator=g).to(dtype);ks=torch.randn((b,s,h*d),generator=g).to(dtype);vs=torch.randn((b,s,h*d),generator=g).to(dtype)
   q4,k4,v4=(x.reshape(b,x.shape[1],h,d).transpose(1,2) for x in [qs,ks,vs])
   mask=None
   if kind=='bool':mask=torch.rand((t,s),generator=g)>.35;mask[0]=False
   elif kind=='bias':mask=torch.randn((b,1,t,s),generator=g)*.1;mask[:,:,0]=-torch.inf
   out=attention(qs.to(args.device),ks.to(args.device),vs.to(args.device),h,mask=None if mask is None else mask.to(args.device),attn_precision=torch.float32).cpu()
   expected=reference(q4,k4,v4,mask).transpose(1,2).reshape(b,t,-1)
   atol,rtol=(3e-5,1e-4) if dtype==torch.float32 else (.003,.002)
   torch.testing.assert_close(out.double(),expected,atol=atol,rtol=rtol)
   records.append(dict(edge='default-tile-tails',dtype=str(dtype),mask=kind,max_abs_error=float((out.double()-expected).abs().max()),passed=True))
 # Unequal value depth, empty token axes, and overflow-sensitive FP16 upcast.
 for t,s,dv in [(0,7,16),(5,0,16),(7,11,8)]:
  q=torch.randn((1,2,t,16),generator=g);k=torch.randn((1,2,s,16),generator=g);v=torch.randn((1,2,s,dv),generator=g)
  out=attention(q.to(args.device),k.to(args.device),v.to(args.device),2,skip_reshape=True,skip_output_reshape=True).cpu()
  expected=torch.zeros_like(out,dtype=torch.float64) if not t or not s else reference(q,k,v,None)
  torch.testing.assert_close(out.double(),expected,atol=3e-5,rtol=1e-4)
  records.append(dict(edge='empty-or-value-depth',shape=[t,s,dv],passed=True))
 if args.device=='cuda':
  q=(torch.randn((1,2,11,16),generator=g)*500).half();k=(torch.randn((1,2,17,16),generator=g)*500).half();v=torch.randn((1,2,17,16),generator=g).half()
  out=attention(q.cuda(),k.cuda(),v.cuda(),2,attn_precision=torch.float32,skip_reshape=True,skip_output_reshape=True).cpu()
  torch.testing.assert_close(out.double(),reference(q,k,v,None),atol=.003,rtol=.002)
  assert torch.isfinite(out).all();records.append(dict(edge='fp16-overflow-upcast',passed=True))
# Negative contracts are checked separately from numerical tolerances.
import polaris_attention as pure
bad=[]
a=torch.zeros((1,3,32));b=torch.zeros((1,7,32));c=torch.zeros((1,7,32))
for name,kwargs,error in [('integer-mask',{'mask':torch.ones((3,7),dtype=torch.int64)},TypeError),('mask-shape',{'mask':torch.ones((5,6),dtype=torch.bool)},ValueError),('budget',{'score_limit_bytes':0},ValueError)]:
 try:pure.attention(a,b,c,2,**kwargs)
 except error:bad.append(dict(case=name,passed=True))
 else:raise AssertionError('Invalid attention contract accepted: '+name)
with torch.inference_mode(False),torch.enable_grad():
 grad=torch.zeros((1,3,32),requires_grad=True)
 try:pure.attention(grad,b,c,2)
 except ValueError:bad.append(dict(case='training-rejected-by-standalone-helper',passed=True))
 else:raise AssertionError('Training path was silently accepted')
policy=[]
if args.comfy_root:
 context,_=load_baseline(args.comfy_root)
 context['attention_sub_quad']=lambda *a,**kw:'stock'
 context['model_management']=types.SimpleNamespace(get_free_memory=lambda *a:(8*1024**3,0))
 ns=dispatch(context);ns['_polaris_tiled_attention']=lambda *a,**kw:'tiled'
 def fake(t,s=8,dtype=torch.float32,grad=False,device='cuda'):
  return types.SimpleNamespace(shape=(1,8,t,64),ndim=4,dtype=dtype,requires_grad=grad,device=types.SimpleNamespace(type=device))
 for name,t,kcount,dtype,mask,grad,dont,expected in [
  ('small',77,77,torch.float32,None,False,False,'stock'),
  ('large',1024,1024,torch.float32,None,False,False,'tiled'),
  ('masked',77,77,torch.float32,True,False,False,'tiled'),
  ('half',77,77,torch.float16,None,False,False,'tiled'),
  ('explicit-no-upcast',77,77,torch.float16,None,False,True,'stock'),
  ('training',1024,1024,torch.float32,None,True,False,'stock')]:
  ns['args'].dont_upcast_attention=dont
  result=ns['attention_sub_quad'](fake(t,dtype=dtype,grad=grad),fake(kcount,dtype=dtype),fake(kcount,dtype=dtype),8,mask=mask,skip_reshape=True)
  assert result==expected,name;policy.append(dict(case=name,selected=result,passed=True))
 # Explicit SDPA/flash/split selection must not be replaced by a sub-quad hook.
 explicit=lambda *a,**kw:None;ct=dict(context);ct['optimized_attention']=explicit;ct['optimized_attention_masked']=explicit
 ct['__name__']='attention_explicit_fixture';ct['__package__']='attention_fixture'
 from attention_patch import HOOK
 exec(compile(HOOK,'<explicit-backend-hook>','exec'),ct)
 assert ct['optimized_attention'] is explicit and ct['optimized_attention_masked'] is explicit
 policy.append(dict(case='explicit-backend-retained',passed=True))
report={'torch':torch.__version__,'device':device_name,'models_used':False,'generation_run':False,'implementation':args.implementation,'guard_checks':bad,'policy_checks':policy,'correctness_cases':len(records),'correctness':records,'correctness_passed':True,'passed':not args.benchmark}
report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'device':device_name,'correctness_cases':len(records),'correctness_passed':True}),flush=True)

if args.benchmark:
 namespace,hashes=load_baseline(args.comfy_root)
 stock=namespace['attention_sub_quad'];report['baseline_source_sha256']=hashes
 bench=[]
 with torch.inference_mode():
  for h,t,s,d in [(8,77,77,64),(8,256,77,64),(8,512,512,64),(8,1024,1024,64),(8,4096,4096,40),(3,513,1025,32),(8,1024,77,64)]:
   q=torch.randn((1,t,h*d),generator=g).cuda();k=torch.randn((1,s,h*d),generator=g).cuda();v=torch.randn((1,s,h*d),generator=g).cuda()
   routed=dispatch(namespace)['attention_sub_quad']
   functions={'stock':stock,'adapted':routed}
   for name,fn in functions.items():
    x=fn(q,k,v,h,attn_precision=torch.float32);torch.cuda.synchronize();del x
   samples={name:[] for name in functions};peaks={name:[] for name in functions}
   for order in ([['stock','adapted'],['adapted','stock']]*4+[['stock','adapted']]):
    for name in order:
     fn=functions[name];torch.cuda.empty_cache();torch.cuda.synchronize();current=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
     start=time.perf_counter();out=fn(q,k,v,h,attn_precision=torch.float32);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
     samples[name].append(ms);peaks[name].append(torch.cuda.max_memory_allocated()-current)
     if name=='stock':expected=out.cpu()
     else:torch.testing.assert_close(out.cpu(),expected,atol=3e-5,rtol=1e-4)
     del out
   row={'shape':[1,h,t,s,d],'samples_ms':samples,'peak_extra_bytes':peaks,'median_ms':{n:statistics.median(v) for n,v in samples.items()},'max_extra_bytes':{n:max(v) for n,v in peaks.items()}}
   bench.append(row);report['benchmark']=bench;report_path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(row),flush=True)
   del q,k,v

if args.benchmark:
 report['benchmark_completed']=True
 report['passed']=True
 report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
