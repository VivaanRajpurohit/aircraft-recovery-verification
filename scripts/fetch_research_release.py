"""Download, checksum and safely extract the frozen research release.

Standard library only. Run in the repository root. Existing files must match;
the extractor refuses to silently replace a different experiment artifact.
"""
from pathlib import Path
import hashlib,json,urllib.request,zipfile,shutil
TAG='v2.1b-research-audit-2026-11-15'
BASE=f'https://github.com/VivaanRajpurohit/aircraft-recovery-verification/releases/download/{TAG}/'
root=Path('.').resolve();cache=root/'.research_downloads';cache.mkdir(exist_ok=True)
manifest=json.loads((root/'artifacts/publication_release/release_assets.json').read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
for a in manifest['assets']:
    p=cache/a['file']
    if not p.exists():
        print('Downloading',a['file'],flush=True);urllib.request.urlretrieve(BASE+a['file'],p)
    if digest(p)!=a['sha256']:raise ValueError('Checksum mismatch: '+a['file'])
    with zipfile.ZipFile(p) as z:
        for info in z.infolist():
            target=(root/info.filename).resolve()
            if not target.is_relative_to(root):raise ValueError('Unsafe archive path')
            if info.is_dir():continue
            if target.exists():
                h=hashlib.sha256()
                with z.open(info) as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
                if digest(target)!=h.hexdigest():raise FileExistsError('Different existing artifact: '+info.filename)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.extracting')
                with z.open(info) as source,temporary.open('wb') as output:shutil.copyfileobj(source,output,8*1024*1024)
                temporary.replace(target)
    print('Verified and extracted',a['file'],flush=True)
