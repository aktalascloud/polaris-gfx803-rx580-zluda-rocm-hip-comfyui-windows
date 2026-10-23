# SPDX-License-Identifier: GPL-3.0-only
import code
import json
import os
from pathlib import Path
import runpy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_devices import radeon, torch_radeon
import torch
import torch_directml
root = Path(__file__).resolve().parent.parent
profile = json.loads((root/'profile.json').read_text(encoding='utf-8-sig'))
selected = radeon(int(os.environ.get('AI_DXGI_DEVICE', profile.get('dxgi_device_id', -1))))
index = os.environ.get('AI_TORCH_DML_DEVICE', profile.get('torch_dml_device_id'))
adapter = torch_radeon(torch_directml, selected, None if index is None else int(index))
dml_device = torch_directml.device(adapter['torch_index'])
scope = {'__name__': '__main__', 'torch': torch, 'torch_directml': torch_directml, 'dml_device': dml_device}
if len(sys.argv) == 1:
    code.interact(local=scope)
elif sys.argv[1] == '-c':
    if len(sys.argv) < 3:
        raise SystemExit('-c requires Python code')
    source = sys.argv[2]; sys.argv = ['-c'] + sys.argv[3:]
    exec(compile(source, '<string>', 'exec'), scope)
elif sys.argv[1] == '-m':
    if len(sys.argv) < 3:
        raise SystemExit('-m requires a module')
    module = sys.argv[2]; sys.argv = sys.argv[2:]
    sys.path.insert(0,os.getcwd())
    runpy.run_module(module, run_name='__main__', init_globals=scope, alter_sys=True)
else:
    script = Path(sys.argv[1]).resolve(); sys.argv = sys.argv[1:]
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__', init_globals=scope)
