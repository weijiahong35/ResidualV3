"""Fetch each frozen-target hidden once, broadcast two bounded RAM-only packs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from queue import Empty
import sys
import time
import torch
from runtime.utils import dump,now
from dp_teacher import install
import runtime.vllm_hidden as hidden

def run_broker(root,queues,acks,children):
    p=json.loads((root/'protocol.json').read_text());torch.set_num_threads(4)
    source=[json.loads(x) for x in Path(p['data_source']).read_text().splitlines()]
    base=Path(__file__).resolve().parent.parent/'teacher_count_head_parallel_pilot_20261007_v1'
    modules={}
    for name in ['train','pipeline']:
        spec=importlib.util.spec_from_file_location('shared_broker_'+name,base/(name+'.py'));module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);modules[name]=module
    packs=modules['train'].packs_for_epoch(source,p['experiment_seed'],2,65536)[:p['pack_count']]
    assert len(packs)==p['pack_count']
    dump(root/'packs.json',dict(indices=packs,same_data_both_arms=True,anchor_seeds=[p['experiment_seed']+200299+i for i in range(len(packs))]))
    from shared_pool import SharedPool,gate
    dump(root/'shared_pool_gate.json',gate(modules['pipeline']))
    pool=SharedPool(slots=2)
    e=json.loads((root/'teacher/endpoint.json').read_text());os.environ['THROUGHPUT_REQUESTS']=str(p['request_concurrency']);os.environ['THROUGHPUT_READERS']=str(p['read_concurrency']);os.environ['THROUGHPUT_KEEPALIVE']='1' if p['http_keepalive'] else '0';install(root/'hidden_requests.jsonl',e)
    provider=hidden.OnlineHiddenProvider(e['endpoint'],e['root'],p['target_model'],prefetch=16)
    provider.begin_epoch([source[index] for indices in packs for index in indices])
    pending={};updated={arm:set() for arm in queues};events=[];next_pack=0
    def handle(v):
        if v['event']=='failed':raise RuntimeError(v)
        if v['event']=='consumed':
            pending[v['pack']]['consumed'].add(v['arm'])
            if len(pending[v['pack']]['consumed'])==len(queues):
                pool.release(pending[v['pack']]['slot']);del pending[v['pack']]
        elif v['event']=='updated':updated[v['arm']].add(v['pack'])
        events.append(v)
        with (root/'consumer_events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    def check():
        for arm,proc in children.items():
            if not proc.is_alive() and not (root/arm/'worker_complete.json').exists():raise RuntimeError(f'{arm} learner exited {proc.exitcode}')
    started=time.time();blocked=0.
    try:
        while next_pack<len(packs) or pending or any(len(s)<len(packs) for s in updated.values()):
            check()
            while True:
                try:handle(acks.get_nowait())
                except Empty:break
            if next_pack<len(packs) and len(pending)<2:
                indices=packs[next_pack];rows=[source[i] for i in indices];begin=time.perf_counter()
                slot,host=pool.acquire(rows)
                fields=['ids','docs','positions','eligible','aux','final']
                allocated=time.perf_counter();modules['pipeline'].fill_host(host,rows,provider);filled=time.perf_counter()
                meta=dict(input_tokens=sum(len(row['input_ids']) for row in rows),examples=len(rows),packed_ids_sha256=hashlib.sha256(host.ids.numpy().tobytes()).hexdigest())
                payload=dict(pack=next_pack+1,metadata=meta,**{k:getattr(host,k) for k in fields})
                pending[next_pack+1]=dict(payload=payload,consumed=set(),slot=slot);assert len(pending)<=2
                for queue in queues.values():queue.put(payload,timeout=300)
                record=dict(pack=next_pack+1,allocate_shared_seconds=allocated-begin,fetch_fill_seconds=filled-allocated,producer_wall_seconds=filled-begin,shared_pack_slots=len(pending),hidden_queried_once=True,**meta,utc=now())
                with (root/'producer_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                dump(root/'status.json',dict(status='shared_hidden_benchmark',produced_pack=next_pack+1,updated={a:len(s) for a,s in updated.items()},utc=now()))
                print(now(),'SHARED_PACK',next_pack+1,'fetch_fill',round(record['fetch_fill_seconds'],3),'slots',len(pending),flush=True)
                next_pack+=1;del host,payload
            else:
                begin=time.perf_counter()
                try:handle(acks.get(timeout=.2))
                except Empty:pass
                blocked+=time.perf_counter()-begin
        dump(root/'broker_complete.json',dict(status='complete',packs=next_pack,updated={a:len(s) for a,s in updated.items()},bounded_shared_pack_count=2,wall_seconds=time.time()-started,ack_wait_seconds=blocked,utc=now()))
    finally:
        provider.close();assert provider.requests==provider.deleted
        dump(root/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,expected_requests=sum(len(indices) for indices in packs),once_per_input_row=True,no_persistent_hidden=True,utc=now()))
        pending.clear()
