# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import code
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_runtime import torch_profile
torch=torch_profile()
if len(sys.argv)==1:
    code.interact(local={'torch':torch})
elif sys.argv[1]=='-c':
    command=sys.argv[2];sys.argv=['-c']+sys.argv[3:]
    exec(compile(command,'<string>','exec'),{'__name__':'__main__','torch':torch})
elif sys.argv[1]=='-m':
    module=sys.argv[2];sys.argv=sys.argv[2:]
    runpy.run_module(module,run_name='__main__',alter_sys=True)
else:
    path=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:]
    sys.path.insert(0,str(path.parent));runpy.run_path(str(path),run_name='__main__')
