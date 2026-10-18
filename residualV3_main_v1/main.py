"""User entry point for the standalone repair project."""
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY=Path('/data/weijiahong/miniconda3/envs/dflash/bin/python')

def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='command',required=True)
    for name in ('prepare','train','verify'):
        q=sp.add_parser(name);q.add_argument('--output',type=Path,required=True)
    q=sp.add_parser('evaluate');q.add_argument('--output',type=Path,required=True);q.add_argument('--gpu',type=int,required=True);q.add_argument('--concurrency',type=int,choices=[1,40],required=True);q.add_argument('--smoke',action='store_true')
    sp.add_parser('export-native')
    a=ap.parse_args();env=os.environ.copy();env.update(PYTHONPATH=f'{ROOT}:{ROOT}/src:{ROOT}/hs_connectors/src',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    def call(module,*args):subprocess.run([str(PY),'-B','-m',module,*map(str,args)],cwd=ROOT,env=env,check=True)
    if a.command=='export-native':
        subprocess.run([str(PY),'-B','-c','from repair.protocol import native_protocol; native_protocol()'],cwd=ROOT,env=env,check=True);return
    out=a.output.resolve()
    if a.command=='evaluate':
        # Export explicitly once before paired smoke/formal to freeze source lineage.
        module='repair.evaluate' if a.concurrency==1 else 'evaluation_c40.evaluate'
        args=['--training',ROOT/'assets/native','--output',out,'--gpu',a.gpu,'--concurrency',a.concurrency,'--prompts',1 if a.smoke and a.concurrency==1 else 20 if a.concurrency==1 else 300,'--tokens',128 if a.smoke else 4096,'--rounds',2]
        if a.smoke:args+=['--smoke']
        elif not (out.parent/(out.name+'_smoke')/'summary.json').exists():
            smoke_args=['--training',ROOT/'assets/native','--output',out.parent/(out.name+'_smoke'),'--gpu',a.gpu,'--concurrency',a.concurrency,'--prompts',1 if a.concurrency==1 else 300,'--tokens',128,'--rounds',1,'--smoke']
            call(module,*smoke_args)
        call(module,*args);return
    if a.command=='verify':call('verify_bundle','--output',out);return
    call('repair.protocol','--output',out)
    if a.command=='train':
        call('repair.validate','--protocol',out/'preregistration.json','--output',out/'cpu_validation.json','--device','cpu')
        call('repair.pipeline','--output',out)
if __name__=='__main__':main()
