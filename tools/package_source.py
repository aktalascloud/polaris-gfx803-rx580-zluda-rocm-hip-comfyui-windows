# SPDX-License-Identifier: GPL-3.0-only
"""Build a source-only archive from a reviewed allowlist; never collect runtimes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

root=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
manifest=json.loads((root/'profiles/source-files.json').read_text(encoding='utf-8'))
allowed={'.md','.txt','.json','.py','.ps1','.cmd','.c','.hip','.def','.diff'}
files=[]
for name in manifest['files']:
    if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Non-relative manifest entry')
    raw=root/name
    for item in [raw,*raw.parents]:
        if item==root:break
        if item.is_symlink() or getattr(item.lstat(),'st_file_attributes',0)&0x400:raise ValueError('Reparse source entry')
    path=raw.resolve(strict=True)
    if not path.is_relative_to(root):raise ValueError('Unsafe source entry')
    if path.suffix not in allowed and path.name not in {'LICENSE','.gitignore','.gitattributes'}:raise ValueError('Non-source entry: '+name)
    if any(part in {'work','runtime','logs','cache','__pycache__','.git'} for part in Path(name).parts):raise ValueError('Private directory')
    text=path.read_text(encoding='utf-8-sig')
    for pattern in [r'(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}',r'hf_[A-Za-z0-9]{24,}',r'sk-[A-Za-z0-9_-]{24,}',r'(?i)\b[A-Z]:[\\/]+Users[\\/]+',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']:
        if re.search(pattern,text):raise ValueError('Sensitive content: '+name)
    files.append((path,name))
output=Path(args.output).resolve()
if output.exists():raise ValueError('Choose a new archive output')
output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for path,name in files:archive.write(path,name)
print(json.dumps({'files':len(files),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'source_only':True}))
