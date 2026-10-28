# SPDX-License-Identifier: GPL-3.0-only
"""Load pinned ComfyUI attention math only; do not start ComfyUI or load weights."""
import ast,hashlib,importlib.util,sys,types
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tools'))
from attention_patch import HOOK,MARKER,UPSTREAM_SHA256,normalized_digest

def load(folder):
 folder=Path(folder).resolve();management=folder/'comfy/model_management.py';file=folder/'comfy/ldm/modules/attention.py';sq=folder/'comfy/ldm/modules/sub_quadratic_attention.py'
 text=file.read_text(encoding='utf-8-sig').replace('\r\n','\n').split(MARKER)[0]
 if normalized_digest(text)!=UPSTREAM_SHA256:raise ValueError('Comparison requires the pinned upstream attention source')
 namespace={'torch':torch,'directml_enabled':False,'is_intel_xpu':lambda:False,'is_ascend_npu':lambda:False,'is_mlu':lambda:False}
 tree=ast.parse(management.read_text(encoding='utf-8-sig'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_free_memory')
 exec(compile(ast.Module(body=[fn],type_ignores=[]),'<pinned-free-memory>','exec'),namespace)
 mm=types.SimpleNamespace(get_free_memory=namespace['get_free_memory'],OOM_EXCEPTION=torch.cuda.OutOfMemoryError)
 old=sys.modules.get('comfy');sys.modules['comfy']=types.SimpleNamespace(model_management=mm)
 try:
  spec=importlib.util.spec_from_file_location('pinned_subquadratic',sq);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 finally:
  if old is None:sys.modules.pop('comfy',None)
  else:sys.modules['comfy']=old
 tree=ast.parse(text);fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['get_attn_precision','attention_sub_quad']]
 context={'torch':torch,'model_management':mm,'efficient_dot_product_attention':module.efficient_dot_product_attention,'args':types.SimpleNamespace(dont_upcast_attention=False),'FORCE_UPCAST_ATTENTION_DTYPE':None}
 exec(compile(ast.Module(body=fns,type_ignores=[]),'<pinned-attention>','exec'),context)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [file,sq,management]}
 return context,hashes

def dispatch(context):
 import polaris_attention
 ns=dict(context);stock=context['attention_sub_quad']
 ns.update(optimized_attention=stock,optimized_attention_masked=stock,__name__='attention_fixture',__package__='attention_fixture')
 package=types.ModuleType('attention_fixture');package.__path__=[]
 sys.modules['attention_fixture']=package;sys.modules['attention_fixture.polaris_attention']=polaris_attention
 exec(compile(HOOK,'<installed-hook>','exec'),ns)
 assert ns['optimized_attention'] is ns['attention_sub_quad'] and ns['optimized_attention_masked'] is ns['attention_sub_quad']
 return ns
