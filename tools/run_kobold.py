# SPDX-License-Identifier: GPL-3.0-only
import ctypes
from pathlib import Path
import os
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_runtime import ROOT,HANDLES,configure
backend=configure()
folder=ROOT/'apps'/'KoboldCPP'
HANDLES.append(os.add_dll_directory(str(folder)))
if backend=='AMD':
    HANDLES.append(ctypes.WinDLL(str(folder/'cublas64_11.dll')))
else:
    cuda_library=ROOT/'envs'/'cuda'/'Lib'/'site-packages'/'torch'/'lib'
    HANDLES.append(os.add_dll_directory(str(cuda_library)))
    HANDLES.append(ctypes.WinDLL(str(cuda_library/'cublas64_11.dll')))
os.chdir(folder)
sys.path.insert(0,str(folder))
sys.argv=[str(folder/'koboldcpp.py')]+sys.argv[1:]
runpy.run_path(str(folder/'koboldcpp.py'),run_name='__main__')
