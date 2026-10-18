"""Bounded real-token hidden alignment check before connecting a second teacher."""
import argparse,json,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parent;RUNTIME=CODE.parent/'runtime_bundle';sys.path[:0]=[str(CODE),str(RUNTIME),str(CODE.parent/'packed20k_recovery_v1')]
from runtime.utils import dump,now

def main():
    import torch
    from transport import install as transport
    from multi_teacher import install
    import runtime.vllm_hidden as hidden
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--extra-service',type=Path,required=True);ap.add_argument('--gpu',type=int,required=True);a=ap.parse_args()
    p=json.loads((a.run/'training_arms/CE_DPACE/training_protocol.json').read_text());e=json.loads((a.extra_service/'endpoint.json').read_text())
    replicas=[dict(endpoint=f"http://127.0.0.1:{p['port']}/v1",root=p['transfer_root']),dict(endpoint=e['endpoint'],root=e['root'])]
    transport(a.output/'check_connection_retries.jsonl');install(a.output/'check_hidden_profile.jsonl',replicas);torch.set_num_threads(4)
    provider=hidden.OnlineHiddenProvider(replicas[0]['endpoint'],replicas[0]['root'],p['target_model'],prefetch=16)
    from device import assignment
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=assignment(a.gpu)['uuid']
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    target=hidden.load_target_components(p['target_model'],torch.device('cuda:0'),replicas[0]['endpoint'],replicas[0]['root'],16)
    rows=[]
    for line in Path(p['data_source']).open():
        row=json.loads(line)
        if 128<=len(row['input_ids'])<=512:rows.append(row)
        if len(rows)==3:break
    assert len(rows)==3;checks=[]
    try:
        for i,row in enumerate(rows):
            x=provider.fetch_from(row,0);y=provider.fetch_from(row,1)
            xf=x.double().flatten();yf=y.double().flatten();difference=xf-yf
            relative=float(difference.norm()/xf.norm().clamp_min(1e-12));cosine=float(torch.nn.functional.cosine_similarity(xf,yf,dim=0))
            check=dict(index=i,tokens=len(row['input_ids']),shape=list(x.shape),dtype=str(x.dtype),bitwise_equal=torch.equal(x,y),relative_l2=relative,
                       cosine=cosine,max_abs=float(difference.abs().max()),passed=relative<.02 and cosine>.999)
            with torch.no_grad():
                indices=torch.linspace(0,len(row['input_ids'])-1,8).round().long()
                lastx=target.model.norm(x[indices,-1].to('cuda'));lasty=target.model.norm(y[indices,-1].to('cuda'))
                zx=lastx@target.lm_head.weight.T;zy=lasty@target.lm_head.weight.T
                px=zx.float().softmax(-1);py=zy.float().softmax(-1)
                tv=.5*(px-py).abs().sum(-1)
                check.update(teacher_label_match=int((zx.argmax(-1)==zy.argmax(-1)).sum()),teacher_label_positions=8,teacher_tv_mean=float(tv.mean()),teacher_tv_max=float(tv.max()))
                check['passed']=check['passed'] and check['teacher_label_match']==8 and check['teacher_tv_max']<.02
            checks.append(check);print(check,flush=True)
        # Exercise the actual concurrent ordered provider, not only direct requests.
        provider.begin_epoch(rows*4);returned=[provider(row) for row in rows*4]
        assert len(returned)==12 and all(t.shape==(len(row['input_ids']),6,4096) for t,row in zip(returned,rows*4))
        assert all(n>0 for n in provider.replica_requests)
        assert all(c['passed'] for c in checks)
        dump(a.output/'teacher_alignment.json',dict(status='passed',checks=checks,ordered_concurrent_requests_passed=True,
              per_replica_requests=provider.replica_requests,per_replica_deleted=provider.replica_deleted,
              endpoint_fallbacks=provider.fallbacks,replicas=replicas,gate=dict(relative_l2=.02,cosine=.999),
              scope='three short real documents plus 12 ordered concurrent requests; not universal or bitwise parity proof',utc=now()))
    finally:
        provider.close();target.hidden_provider.close()

if __name__=='__main__':main()
