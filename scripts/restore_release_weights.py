"""Restore final E1/E2 gzip assets from local downloads or a public Release."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

class DigestReader:
    def __init__(self,source):
        self.source=source;self.h=hashlib.sha256();self.count=0
    def read(self,n=-1):
        b=self.source.read(n);self.h.update(b);self.count+=len(b);return b

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8*2**20):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,help='Directory containing downloaded .safetensors.gz files')
    parser.add_argument('--repo',help='Public GitHub owner/repository for direct download')
    parser.add_argument('--tag',default='v2.0-archive')
    parser.add_argument('--epoch',choices=['E1','E2'])
    args=parser.parse_args()
    manifest=ROOT/'WEIGHTS_MANIFEST.json'
    if not manifest.is_file():parser.error('Use this tool inside the extracted release project containing WEIGHTS_MANIFEST.json')
    spec=json.loads(manifest.read_text())
    if args.repo and not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',args.repo):parser.error('Invalid owner/repository')
    for e in spec['assets']:
        if args.epoch and e['epoch']!=args.epoch:continue
        target=(ROOT/e['restored_path']).resolve()
        if not target.is_relative_to(ROOT) or target==ROOT:raise RuntimeError('Invalid restore path')
        if target.exists():
            if target.is_file() and target.stat().st_size==e['restored_bytes'] and sha(target)==e['restored_sha256']:
                print('verified existing:',e['restored_path']);continue
            raise RuntimeError('Existing target differs; refusing to overwrite: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(target.parent).free<e['restored_bytes']+2**30:raise RuntimeError('Insufficient disk space')
        local=args.archives/e['name'] if args.archives else None
        if local and local.is_file():source=local.open('rb')
        elif args.repo:
            url=f'https://github.com/{args.repo}/releases/download/{urllib.parse.quote(args.tag,safe="")}/{urllib.parse.quote(e["name"],safe="")}'
            source=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResidualV3-weight-restore'}),timeout=120)
        else:parser.error('Provide --archives DIRECTORY or --repo OWNER/REPOSITORY')
        partial=target.with_name(target.name+'.partial')
        output_hash=hashlib.sha256();count=0
        with source:
            packed=DigestReader(source)
            with partial.open('xb') as output,gzip.GzipFile(fileobj=packed,mode='rb') as decoded:
                while b:=decoded.read(8*2**20):
                    count+=len(b)
                    if count>e['restored_bytes']:raise RuntimeError('Unexpected expanded size')
                    output.write(b);output_hash.update(b)
            if (packed.count!=e['bytes'] or packed.h.hexdigest()!=e['sha256'] or
                    count!=e['restored_bytes'] or output_hash.hexdigest()!=e['restored_sha256']):
                raise RuntimeError('Checksum mismatch; partial retained: '+str(partial))
        partial.replace(target)
        print('verified and restored:',e['restored_path'])

if __name__=='__main__':main()
