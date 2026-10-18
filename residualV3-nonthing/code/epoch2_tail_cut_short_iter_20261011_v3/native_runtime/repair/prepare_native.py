import argparse,json
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from .protocol import read
from .model import DualFeedbackRepair,ARMS
from .run import fresh_head
from runtime.utils import dump,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--initialization-smoke',action='store_true');a=ap.parse_args();out=a.output.resolve()
    p=read(out/'preregistration.json');native=out/('initialization_native' if a.initialization_smoke else 'native');native.mkdir(exist_ok=False)
    if a.initialization_smoke:p['validation_only_untrained']=True
    else:
        summary=json.loads((out/'training/summary.json').read_text())
        assert summary['status']=='complete' and summary['steps']==p['train_count']*p['epochs'] and summary['repair_initialization']=='epoch2_weights_optimizer_reset'
    ck={}
    for arm in ARMS:
        if a.initialization_smoke:
            state={k:v.detach().cpu().contiguous().clone() for k,v in fresh_head(p).to(dtype=torch.bfloat16).state_dict().items()}
            original=native/f'{arm}_random_fixture.safetensors';save_file(state,str(original));source=dict(path=str(original),sha256=sha(original))
        else:
            source=summary['checkpoints'][arm]['3'];assert sha(source['path'])==source['sha256']
            state=load_file(source['path']);head=DualFeedbackRepair(**p['architecture']);head.load_state_dict(state,strict=True)
        dst=native/f'{arm}.safetensors';save_file({'head.'+k:v for k,v in state.items()},str(dst))
        saved=load_file(str(dst));assert all(torch.equal(v,saved['head.'+k]) for k,v in state.items())
        ck[arm]=dict(path=str(dst),sha256=sha(dst),original=source['path'],original_sha256=source['sha256'],head_config_path=p['head_config_path'],head_config_sha256=sha(p['head_config_path']))
        p['asset_sha256'].update({str(dst):sha(dst),source['path']:source['sha256']})
        if not a.initialization_smoke:
            f=out/'training'/arm/'initialization.json';p['asset_sha256'][str(f)]=sha(f)
    p.update(native_arms=['RAW',*ARMS],native_configs={arm:{} for arm in ARMS},native_checkpoints=ck)
    dump(native/'preregistration.json',p)
if __name__=='__main__':main()
