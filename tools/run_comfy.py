# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import os
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_runtime import ROOT,torch_profile
torch_profile()
folder=ROOT/'apps'/'ComfyUI'
os.chdir(folder)
sys.path.insert(0,str(folder))
sys.argv=[str(folder/'main.py')]+sys.argv[1:]
runpy.run_path(str(folder/'main.py'),run_name='__main__')
