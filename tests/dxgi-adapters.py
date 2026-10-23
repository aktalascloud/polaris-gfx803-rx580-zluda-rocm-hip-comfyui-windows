# SPDX-License-Identifier: GPL-3.0-only
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent.parent/'tools'/'gpu_devices.py'),run_name='__main__')
