# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import code
import os
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_runtime import torch_profile
torch=torch_profile()
if len(sys.argv)==1:
    code.interact(local={'torch':torch})
elif sys.argv[1]=='-c':
    if len(sys.argv)<3:raise SystemExit('-c requires Python code')
    command=sys.argv[2];sys.argv=['-c']+sys.argv[3:]
    exec(compile(command,'<string>','exec'),{'__name__':'__main__','torch':torch})
elif sys.argv[1]=='-m':
    if len(sys.argv)<3:raise SystemExit('-m requires a module')
    module=sys.argv[2];sys.argv=sys.argv[2:]
    sys.path.insert(0,os.getcwd())
    runpy.run_module(module,run_name='__main__',alter_sys=True)
else:
    path=Path(sys.argv[1]).resolve();sys.argv=sys.argv[1:]
    sys.path.insert(0,str(path.parent));runpy.run_path(str(path),run_name='__main__')
