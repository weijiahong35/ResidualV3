"""Lazy online data supply and two-phase commit of the common data cursor."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from queue import Empty
import shutil
import sys
import time
import numpy as np
import torch
from data_source import IndexedData
from dp_teacher import install
from full_checkpoint import atomic,commit
from runtime.utils import now
import runtime.vllm_hidden as hidden
from shared_pool import SharedPool,gate


def run_broker(root,queues,acks,children):
    p=json.loads((root/'protocol.json').read_text());job=Path(p['job_root']);torch.set_num_threads(p['producer_torch_threads'])
    source=IndexedData(json.loads(Path(p['data_manifest']).read_text()))
    order=np.load(p['data_order'],mmap_mode='r',allow_pickle=False);bounds=np.load(p['pack_offsets'],mmap_mode='r',allow_pickle=False)
    start=json.loads((root/'start_cursor.json').read_text())['cursor'];total=p['pack_count'];assert len(bounds)==total+1
    spec=importlib.util.spec_from_file_location('production_host_reference',Path(__file__).resolve().parent.parent/'teacher_count_head_parallel_pilot_20261007_v1/pipeline.py')
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    atomic(root/'shared_pool_gate.json',gate(reference));pool=SharedPool(slots=2)
    endpoint=json.loads((root/'teacher/endpoint.json').read_text())
    for key,value in {'THROUGHPUT_REQUESTS':p['request_concurrency'],'THROUGHPUT_READERS':p['read_concurrency'],
        'THROUGHPUT_KEEPALIVE':0,'THROUGHPUT_MMAP_OWNERSHIP':1,'THROUGHPUT_LOCK_POLL':p['lock_poll_seconds'],
        'THROUGHPUT_LOG_EVERY':p['hidden_request_log_every']}.items():os.environ[key]=str(value)
    install(root/'hidden_requests.jsonl',endpoint)
    provider=hidden.OnlineHiddenProvider(endpoint['endpoint'],endpoint['root'],p['target_model'],prefetch=16)
    provider.begin_epoch(source.read_order(order[int(bounds[start]):],workers=8,prefetch=256))
    pending={};updated={arm:start for arm in queues};next_pack=start;used_requests=0
    checkpoint_cursor=None;checkpoint_action=None;descriptors={};last_commit=start
    next_periodic=1 if start==0 else min(total,(start//p['checkpoint_interval']+1)*p['checkpoint_interval'])
    def check():
        for arm,process in children.items():
            if not process.is_alive() and not(root/arm/'worker_complete.json').exists():raise RuntimeError(f'{arm} learner exited {process.exitcode}')
    def handle(v):
        if v['event']=='failed':raise RuntimeError(v)
        if v['event']=='consumed':
            pending[v['pack']]['consumed'].add(v['arm'])
            if len(pending[v['pack']]['consumed'])==len(queues):pool.release(pending[v['pack']]['slot']);del pending[v['pack']]
        elif v['event']=='updated':
            assert v['pack']==updated[v['arm']]+1;updated[v['arm']]=v['pack']
        elif v['event']=='checkpoint_saved':
            assert v['cursor']==checkpoint_cursor;descriptors[v['arm']]=v['descriptor']
        with (root/'consumer_events.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
    try:
        while True:
            check()
            while True:
                try:handle(acks.get_nowait())
                except Empty:break
            free=shutil.disk_usage(job).free
            if free<128*2**20:
                emergency=job/'metadata_emergency.reserve'
                if emergency.exists():emergency.unlink()
                atomic(job/'pause.request.json',dict(reason='disk_metadata_headroom_low',free_bytes=free))
            pause=(job/'pause.request.json').exists();request=(job/'checkpoint.request.json').exists()
            barrier=pause or request or next_pack>=next_periodic or next_pack==total
            if checkpoint_cursor is None and barrier and next_pack==min(updated.values()) and not pending:
                checkpoint_cursor=next_pack;checkpoint_action='complete' if next_pack==total else('pause' if pause else'continue')
                latest=json.loads((job/'latest_checkpoint.json').read_text()) if(job/'latest_checkpoint.json').exists() else None
                slot=1-latest['slot'] if latest else 0;descriptors={}
                for queue in queues.values():queue.put(dict(event='checkpoint',cursor=next_pack,slot=slot))
                atomic(job/'status.json',dict(status='saving_checkpoint',cursor=next_pack,total_packs=total,epochs=1,training_rows=p['training_rows'],action=checkpoint_action,utc=now()))
            if checkpoint_cursor is not None:
                if len(descriptors)==len(queues):
                    pointer=commit(job,descriptors,checkpoint_cursor)
                    for queue in queues.values():queue.put(dict(event='checkpoint_committed',cursor=checkpoint_cursor,action=checkpoint_action))
                    print(now(),'PAIRED_CHECKPOINT_COMMITTED',checkpoint_cursor,'slot',pointer['slot'],flush=True)
                    last_commit=checkpoint_cursor;(job/'checkpoint.request.json').unlink(missing_ok=True)
                    if checkpoint_action in ['complete','pause']:
                        atomic(root/'broker_complete.json',dict(status=checkpoint_action,cursor=next_pack,updated=updated,used_requests=used_requests,utc=now()))
                        return checkpoint_action
                    checkpoint_cursor=None;descriptors={};next_periodic=min(total,(next_pack//p['checkpoint_interval']+1)*p['checkpoint_interval'])
                else:
                    try:handle(acks.get(timeout=.2))
                    except Empty:pass
                continue
            if not barrier and len(pending)<2:
                indices=order[int(bounds[next_pack]):int(bounds[next_pack+1])]
                rows=list(source.read_order(indices,workers=8,prefetch=128));begin=time.perf_counter();slot,host=pool.acquire(rows);allocated=time.perf_counter()
                reference.fill_host(host,rows,provider)
                assert host.offset<=65536 and len(host.eligible)>0
                meta=dict(input_tokens=host.offset,examples=len(rows),packed_ids_sha256=hashlib.sha256(host.ids.numpy().tobytes()).hexdigest(),fully_supervised_anchor_candidates=len(host.eligible))
                payload=dict(event='pack',pack=next_pack+1,metadata=meta,**{k:getattr(host,k) for k in ['ids','docs','positions','eligible','aux','final']})
                pending[next_pack+1]=dict(consumed=set(),slot=slot)
                for queue in queues.values():queue.put(payload,timeout=300)
                used_requests+=len(rows);next_pack+=1
                record=dict(pack=next_pack,allocate_shared_seconds=allocated-begin,fetch_fill_seconds=time.perf_counter()-allocated,
                    producer_wall_seconds=time.perf_counter()-begin,hidden_queried_once=True,**meta,utc=now())
                with (root/'producer_metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if next_pack%10==0:
                    atomic(job/'status.json',dict(status='training',epochs=1,total_packs=total,training_rows=p['training_rows'],produced_pack=next_pack,
                        updated=updated,latest_committed_cursor=last_commit,validation=False,evaluation=False,utc=now()))
                print(now(),'SHARED_PACK',next_pack,'/',total,'fetch_fill',round(record['fetch_fill_seconds'],3),flush=True)
                del rows,host,payload
            else:
                try:handle(acks.get(timeout=.2))
                except Empty:pass
    finally:
        provider.close();assert provider.requests==provider.deleted
        atomic(root/'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,used_requests=used_requests,
            prefetch_may_fetch_unused_rows_on_pause=True,hidden_once_per_used_row=True,no_persistent_hidden_logits=True));pending.clear()
