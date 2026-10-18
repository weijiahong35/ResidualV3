"""Bounded self-repeat and cross-service comparison; persist scalar evidence only."""
import argparse,json,os,sys,itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
CODE=Path(__file__).resolve().parent;BASE=CODE.parent;RUNTIME=BASE/'runtime_bundle'
sys.path[:0]=[str(CODE),str(RUNTIME),str(BASE/'packed20k_recovery_v1'),str(BASE/'fused_output_dpace_dualteacher_v5')]
from runtime.utils import dump,now

def main():
    import torch
    from transport import install as transport
    from multi_teacher import install
    import runtime.vllm_hidden as hidden
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--service',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(exist_ok=False)
    p=json.loads((a.run/'training_arms/CE_DPACE/training_protocol.json').read_text());e=json.loads((a.service/'endpoint.json').read_text())
    replicas=[dict(endpoint=f"http://127.0.0.1:{p['port']}/v1",root=p['transfer_root']),dict(endpoint=e['endpoint'],root=e['root'])]
    transport(a.output/'connection_retries.jsonl');install(a.output/'request_profile.jsonl',replicas);torch.set_num_threads(4)
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False
    free,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(min(free-4*2**30,4*2**30)/total)
    target=hidden.load_target_components(p['target_model'],torch.device('cuda:0'),replicas[0]['endpoint'],replicas[0]['root'],16);provider=target.hidden_provider
    rows=[]
    for line in Path(p['data_source']).open():
        row=json.loads(line)
        if 128<=len(row['input_ids'])<=512:rows.append(row)
        if len(rows)==3:break
    checks=[]
    def pair(x,y,px,py,zx,zy):
        xf=x.double().flatten();yf=y.double().flatten();tv=.5*(px-py).abs().sum(-1)
        return dict(bitwise_equal=torch.equal(x,y),relative_l2=float((xf-yf).norm()/xf.norm().clamp_min(1e-12)),
                    cosine=float(torch.nn.functional.cosine_similarity(xf,yf,dim=0)),max_abs=float((xf-yf).abs().max()),
                    teacher_tv_mean=float(tv.mean()),teacher_tv_max=float(tv.max()),teacher_label_match=int((zx==zy).sum()),teacher_label_positions=8)
    try:
        # Alternate endpoints; then vary scheduling with a twelve-request burst.
        samples=[[[] for _ in rows] for _ in replicas]
        for repeat in range(3):
            for i,row in enumerate(rows):
                for replica in range(2):samples[replica][i].append(provider.fetch_from(row,replica))
        for replica in range(2):
            with ThreadPoolExecutor(max_workers=12) as pool:
                values=list(pool.map(lambda row:provider.fetch_from(row,replica),rows*4))
            for j,value in enumerate(values):samples[replica][j%3].append(value)
        with torch.no_grad():
            for i,row in enumerate(rows):
                probabilities=[[],[]];labels=[[],[]];indices=torch.linspace(0,len(row['input_ids'])-1,8).round().long()
                for replica in range(2):
                    for value in samples[replica][i]:
                        z=target.model.norm(value[indices,-1].to('cuda'))@target.lm_head.weight.T
                        probabilities[replica].append(z.float().softmax(-1).cpu());labels[replica].append(z.argmax(-1).cpu())
                categories={}
                for name,pairs in [('old_self',[(0,j,0,k) for j,k in itertools.combinations(range(7),2)]),
                                   ('dp_self',[(1,j,1,k) for j,k in itertools.combinations(range(7),2)]),
                                   ('old_vs_dp',[(0,j,1,k) for j in range(7) for k in range(7)])]:
                    values=[pair(samples[r][i][j],samples[s][i][k],probabilities[r][j],probabilities[s][k],labels[r][j],labels[s][k]) for r,j,s,k in pairs]
                    categories[name]=dict(pairs=len(values),max_relative_l2=max(v['relative_l2'] for v in values),min_cosine=min(v['cosine'] for v in values),
                        max_teacher_tv=max(v['teacher_tv_max'] for v in values),min_label_match=min(v['teacher_label_match'] for v in values),bitwise_pairs=sum(v['bitwise_equal'] for v in values),measurements=values)
                check=dict(index=i,tokens=len(row['input_ids']),categories=categories);checks.append(check)
                print('REPEAT_CHECK',i,{k:{m:v[m] for m in ['max_relative_l2','min_cosine','max_teacher_tv','min_label_match','bitwise_pairs']} for k,v in categories.items()},flush=True)
        # Exercise the actual single-endpoint DP provider's FIFO path separately.
        provider.close()
        import runtime.vllm_hidden as current
        from dp_teacher import install as install_dp
        from transport import install as install_plain
        # Restore the resilient single provider before wrapping with DP.
        current.OnlineHiddenProvider=hidden.OnlineHiddenProvider.__mro__[1]
        install_dp(a.output/'dp_fifo_profile.jsonl',replicas[1])
        dp=current.OnlineHiddenProvider(e['endpoint'],e['root'],p['target_model'],prefetch=16)
        try:
            dp.begin_epoch(rows*12)
            values=[dp(row) for row in rows*12]
            assert dp.prefetch==32 and all(x.shape==(len(row['input_ids']),6,4096) for x,row in zip(values,rows*12))
        finally:dp.close()
        cleanup=dict(dp_requests=dp.requests,dp_deleted=dp.deleted,dp_inflight_peak=dp.peak_inflight,temporary_files_remaining=[str(x) for x in Path(e['root']).iterdir()])
        strict=all(v['categories']['old_vs_dp']['max_relative_l2']<.02 and v['categories']['old_vs_dp']['min_cosine']>.999 and v['categories']['old_vs_dp']['max_teacher_tv']<.02 and v['categories']['old_vs_dp']['min_label_match']==8 for v in checks)
        # Diagnostic only: report whether cross-service variation is within the
        # OLD service's measured self-repeat envelope; never redefine the old gate.
        envelope=all(v['categories']['old_vs_dp']['max_teacher_tv']<=v['categories']['old_self']['max_teacher_tv']+1e-5 and
                     v['categories']['old_vs_dp']['max_relative_l2']<=v['categories']['old_self']['max_relative_l2']+1e-5 and
                     v['categories']['old_vs_dp']['min_label_match']==8 for v in checks)
        dump(a.output/'summary.json',dict(status='complete',strict_original_gate_passed=strict,cross_within_old_repeat_envelope=envelope,checks=checks,
             ordered_concurrent_requests_passed=True,cleanup=cleanup,replicas=replicas,teacher_source_model=p['target_model'],
             scope='three short real documents, 7 samples per endpoint and 36 FIFO DP requests; not universal equivalence',utc=now()))
    finally:provider.close()
if __name__=='__main__':main()
