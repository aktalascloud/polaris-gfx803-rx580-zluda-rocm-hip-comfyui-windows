# SPDX-License-Identifier: GPL-3.0-only
"""Small synthetic checks only: no model files or generation."""
import argparse
import json
from pathlib import Path
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from gpu_devices import radeon, torch_radeon
parser=argparse.ArgumentParser()
parser.add_argument('--backend',choices=['torch-directml','onnx-directml','openvino'],required=True)
parser.add_argument('--dxgi-device',type=int,default=-1)
parser.add_argument('--torch-device',type=int)
parser.add_argument('--report',required=True)
args=parser.parse_args()
destination=Path(args.report).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
report={'backend':args.backend,'models_downloaded':False,'generation_run':False}
if args.backend=='torch-directml':
    import torch
    import torch_directml as dml
    selected=radeon(args.dxgi_device)
    device_info=torch_radeon(dml,selected,args.torch_device)
    device=dml.device(device_info['torch_index'])
    if device.type=='cpu':raise RuntimeError('CPU is not a DirectML device')
    report.update(torch=torch.__version__,torch_directml=__import__('importlib.metadata',fromlist=['version']).version('torch-directml'),device=device_info,dxgi_index=selected['dxgi_index'],checks=[])
    generator=torch.Generator().manual_seed(580)
    for dtype in [torch.float32,torch.float16]:
        for operation in ['matmul','conv2d']:
            shape_a,shape_b=((31,17),(17,23)) if operation=='matmul' else ((1,3,16,16),(4,3,3,3))
            a=(torch.rand(shape_a,generator=generator)-.5).to(dtype)
            b=(torch.rand(shape_b,generator=generator)-.5).to(dtype)
            op=(lambda a,b:a@b) if operation=='matmul' else (lambda a,b:torch.nn.functional.conv2d(a,b,padding=1))
            reference=op(a.float(),b.float()).to(dtype)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                output=op(a.to(device),b.to(device)).cpu()
            if any('fall' in str(w.message).lower() and 'cpu' in str(w.message).lower() for w in caught):
                raise RuntimeError('A tested Torch operation reported CPU fallback')
            atol,rtol=(3e-5,1e-4) if dtype==torch.float32 else (3e-3,2e-3)
            torch.testing.assert_close(output,reference,atol=atol,rtol=rtol)
            report['checks'].append(dict(dtype=str(dtype),operation=operation,max_abs_error=(output.float()-reference.float()).abs().max().item(),atol=atol,rtol=rtol,passed=True))
elif args.backend=='onnx-directml':
    import numpy as np
    import onnxruntime as ort
    selected=radeon(args.dxgi_device)
    # Tiny ModelProto encoded with standard protobuf wire fields; no onnx package/model download.
    def varint(value):
        data=bytearray()
        while value>127:data.append((value&127)|128);value>>=7
        data.append(value);return bytes(data)
    def integer(field,value):return varint(field<<3)+varint(value)
    def blob(field,value):
        if isinstance(value,str):value=value.encode()
        return varint((field<<3)|2)+varint(len(value))+value
    def info(name,shape):
        tensor=integer(1,1)+blob(2,b''.join(blob(1,integer(1,d)) for d in shape))
        return blob(1,name)+blob(2,blob(1,tensor))
    node=blob(1,'A')+blob(1,'B')+blob(2,'Y')+blob(4,'MatMul')
    graph=blob(1,node)+blob(2,'synthetic-matmul')+blob(11,info('A',[2,3]))+blob(11,info('B',[3,2]))+blob(12,info('Y',[2,2]))
    model=integer(1,9)+blob(7,graph)+blob(8,integer(2,13))
    options=ort.SessionOptions();options.enable_mem_pattern=False;options.enable_profiling=True
    options.profile_file_prefix=str(destination.parent/'onnx-numeric-profile')
    options.add_session_config_entry('session.disable_cpu_ep_fallback','1')
    session=ort.InferenceSession(model,sess_options=options,providers=[('DmlExecutionProvider',{'device_id':str(selected['dxgi_index'])})])
    a=np.arange(6,dtype=np.float32).reshape(2,3)/7;b=np.arange(6,dtype=np.float32).reshape(3,2)/11
    output=session.run(None,{'A':a,'B':b})[0]
    np.testing.assert_allclose(output,a@b,rtol=1e-4,atol=3e-5)
    profile=Path(session.end_profiling())
    events=json.loads(profile.read_text())
    providers={x.get('args',{}).get('provider') for x in events if x.get('cat')=='Node' and x.get('args',{}).get('provider')}
    if providers!={'DmlExecutionProvider'}:raise RuntimeError('Expected strictly DirectML node execution: '+str(providers))
    report.update(version=ort.__version__,dxgi_index=selected['dxgi_index'],device=selected['name'],node_providers=sorted(providers),cpu_fallback_disabled=True,max_abs_error=float(np.max(np.abs(output-a@b))))
else:
    import numpy as np
    import openvino as ov
    from openvino import opset13 as ops
    a=np.arange(6,dtype=np.float32).reshape(2,3)/7;b=np.arange(6,dtype=np.float32).reshape(3,2)/11
    pa,pb=ops.parameter([2,3],np.float32),ops.parameter([3,2],np.float32)
    model=ov.Model([ops.matmul(pa,pb,False,False)],[pa,pb])
    compiled=ov.Core().compile_model(model,'CPU',{'INFERENCE_PRECISION_HINT':'f32'})
    output=next(iter(compiled({0:a,1:b}).values()))
    np.testing.assert_allclose(output,a@b,rtol=1e-4,atol=3e-5)
    report.update(version=ov.__version__,execution_devices=list(compiled.get_property('EXECUTION_DEVICES')),max_abs_error=float(np.max(np.abs(output-a@b))))
report['passed']=True
destination.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
