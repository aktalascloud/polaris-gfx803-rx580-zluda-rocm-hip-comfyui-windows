# SPDX-License-Identifier: GPL-3.0-only
"""Read-only DLL and adapter inventory. Does not initialize CUDA or load Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_devices import adapters
from hip_runtime import describe, resolve
from attention_patch import status as attention_status

parser=argparse.ArgumentParser()
parser.add_argument('--root',default=str(Path(__file__).resolve().parent.parent))
parser.add_argument('--hip-runtime')
parser.add_argument('--report')
args=parser.parse_args();root=Path(args.root).resolve()
profile=json.loads((root/'profile.json').read_text(encoding='utf-8-sig'))
aliases=[]
for source,target in [('cublas.dll','cublas64_11.dll'),('cusparse.dll','cusparse64_11.dll'),('nvrtc.dll','nvrtc64_112_0.dll')]:
    a,b=root/'zluda'/source,root/'envs/cuda/Lib/site-packages/torch/lib'/target
    same=a.exists() and b.exists() and hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
    aliases.append(dict(alias=target,matches_pinned_zluda=same))
original=root/'envs/cuda/Lib/site-packages/torch/lib/nvrtc_cuda.dll'
expected=profile.get('original_nvrtc_sha256')
nvrtc={'preserved':original.exists(),'sha256':hashlib.sha256(original.read_bytes()).hexdigest() if original.exists() else None,'verified_original':(hashlib.sha256(original.read_bytes()).hexdigest()==expected) if original.exists() and expected else None}
receipt=root/'hip/bin/small-gemm-patch.json'
state=json.loads(receipt.read_text(encoding='utf-8-sig')) if receipt.exists() else None
report=dict(hip_runtime=describe(resolve(root,args.hip_runtime)),adapters=adapters(),aliases=aliases,original_nvrtc=nvrtc,small_gemm_patch_enabled=bool(state and state.get('enabled')),attention=attention_status(root/'apps/ComfyUI'),gpu_workload_run=False)
if args.report:Path(args.report).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
