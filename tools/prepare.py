# SPDX-License-Identifier: GPL-3.0-only
"""Filesystem-only installation helpers. No GPU initialization or inference."""
import argparse
from pathlib import Path
import re
import shutil
import zipfile

def under(root, name):
    root=Path(root).resolve()
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Archive path escaped destination')
    return path

def unzip(source,destination):
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            under(destination,item.filename)
        archive.extractall(destination)

def patch_comfy(root,patch):
    # The distributed GPL patch is restricted to these two upstream files.
    lines=Path(patch).read_text(encoding='utf-8-sig').splitlines()
    position=0
    while position<len(lines):
        if not lines[position].startswith('+++ b/'):
            position+=1;continue
        name=lines[position][6:]
        if name not in {'comfy/zluda.py','comfyui.bat'}:raise ValueError('Unexpected patched file')
        path=under(root,name)
        original=path.read_text(encoding='utf-8-sig').splitlines()
        result=[]; consumed=0; position+=1
        while position<len(lines) and not lines[position].startswith('diff --git '):
            match=re.match(r'@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@',lines[position])
            if not match:position+=1;continue
            start=int(match[1])-1
            result.extend(original[consumed:start]);consumed=start;position+=1
            while position<len(lines) and not lines[position].startswith(('@@ ','diff --git ')):
                line=lines[position]
                if line.startswith((' ','-')):
                    if consumed>=len(original) or original[consumed]!=line[1:]:raise ValueError('Patch context mismatch')
                    consumed+=1
                if line.startswith((' ','+')):result.append(line[1:])
                position+=1
        result.extend(original[consumed:])
        path.write_text('\n'.join(result)+'\n',encoding='utf-8')
    (Path(root)/'Start-ComfyUI.cmd').write_text('@echo off\ncall "%~dp0..\\..\\Start-ComfyUI.cmd" %*\n',encoding='ascii')

def prune_comfy(root):
    root=Path(root).resolve()
    for name in ['tests','tests-unit','.github','.ci','script_examples']:
        path=under(root,name)
        if path.exists():shutil.rmtree(path)
    for path in root.glob('*.bat'):
        if path.name!='comfyui.bat':path.unlink()
    for name in ['new_updater.py','testzluda.py','pytest.ini']:
        path=under(root,name)
        if path.is_file():path.unlink()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['unzip','patch-comfy','prune-comfy'])
    parser.add_argument('source');parser.add_argument('destination',nargs='?')
    args=parser.parse_args()
    if args.operation=='unzip':unzip(args.source,args.destination)
    elif args.operation=='patch-comfy':patch_comfy(args.source,args.destination)
    else:prune_comfy(args.source)
