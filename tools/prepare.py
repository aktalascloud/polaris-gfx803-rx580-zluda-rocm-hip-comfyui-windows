# SPDX-License-Identifier: GPL-3.0-only
"""Filesystem-only installation helpers. No GPU initialization or inference."""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import struct
import zipfile
import zlib

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

def extract_kobold(source,destination):
    """Read the public PyInstaller CArchive; never execute the downloaded EXE."""
    source=Path(source);destination=Path(destination)
    size=source.stat().st_size
    cookie=struct.Struct('!8sIIII64s')
    magic=b'MEI\014\013\012\013\016'
    with source.open('rb') as stream:
        stream.seek(max(0,size-1024*1024));tail=stream.read()
        relative=tail.rfind(magic)
        if relative<0:raise ValueError('PyInstaller cookie is missing')
        cookie_position=max(0,size-1024*1024)+relative
        _,package_length,toc_offset,toc_size,_,_=cookie.unpack_from(tail,relative)
        start=cookie_position+cookie.size-package_length
        stream.seek(start+toc_offset);toc=stream.read(toc_size)
        position=0;found=set()
        names={'koboldcpp_cublas.dll','cudart64_110.dll','kcpp_docs.embd','kcpp_sdui.embd','klite.embd',
               'LICENSE.md','MIT_LICENSE_GGML_SDCPP_LLAMACPP_ONLY.md'}
        while position<len(toc):
            entry_length,offset,packed,unpacked,compressed,kind=struct.unpack_from('!iIIIBc',toc,position)
            if entry_length<18 or position+entry_length>len(toc):raise ValueError('Invalid archive table')
            name=toc[position+18:position+entry_length].rstrip(b'\0').decode('utf-8')
            if name in names or name.startswith(('kcpp_adapters/','kcpp_adapters\\')):
                target=under(destination,name.replace('\\','/'))
                stream.seek(start+offset);data=stream.read(packed)
                if compressed:data=zlib.decompress(data)
                if len(data)!=unpacked:raise ValueError('Extracted file length mismatch')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);found.add(name)
            position+=entry_length
    if not {'koboldcpp_cublas.dll','cudart64_110.dll','klite.embd'}.issubset(found):raise ValueError('Required Kobold files missing')
    expected='3e089065368bb5710887656309456f3eacf971423ebb92da6d31d827d7e2946a'
    if hashlib.sha256((destination/'koboldcpp_cublas.dll').read_bytes()).hexdigest()!=expected:
        raise ValueError('Kobold native DLL does not match the measured version')

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
    parser.add_argument('operation',choices=['unzip','patch-comfy','extract-kobold','prune-comfy'])
    parser.add_argument('source');parser.add_argument('destination',nargs='?')
    args=parser.parse_args()
    if args.operation=='unzip':unzip(args.source,args.destination)
    elif args.operation=='patch-comfy':patch_comfy(args.source,args.destination)
    elif args.operation=='extract-kobold':extract_kobold(args.source,args.destination)
    else:prune_comfy(args.source)
