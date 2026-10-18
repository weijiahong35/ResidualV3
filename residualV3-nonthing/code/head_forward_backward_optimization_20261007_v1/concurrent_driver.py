"""Consume identical shared packs with independent weights and optimizers."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from runtime.utils import dump,now
from joint_runner import update,save
from host_pipeline import upload

def run_worker(root,arm,p,target,pair,opt,basis,in_queue,acks):
    from registered_storage import RegisteredStoragePool
    with RegisteredStoragePool() as registrations:
        return _run_worker(root,arm,p,target,pair,opt,basis,in_queue,acks,registrations)

def _run_worker(root,arm,p,target,pair,opt,basis,in_queue,acks,registrations):
    out=root/arm;records=[]
    for i in range(p['pack_count']):
        begin=time.perf_counter();payload=in_queue.get(timeout=300)
        received=time.perf_counter();assert payload['pack']==i+1
        meta=payload['metadata'];fields=['ids','docs','positions','eligible','aux','final']
        assert all(payload[k].is_shared() for k in fields)
        pin_begin=time.perf_counter()
        pinned={k:registrations.pin(payload[k]) for k in fields}
        pin_seconds=time.perf_counter()-pin_begin
        host=SimpleNamespace(**pinned,offset=meta['input_tokens'])
        device,transfer=upload(host,target)
        if i==0:
            assert torch.equal(device['aux'].cpu(),host.aux),'registered aux H2D mismatch'
            with torch.no_grad():reference_last=target.model.norm(host.final.to('cuda'))
            assert torch.equal(device['last'],reference_last),'target norm after registered H2D mismatch'
            del reference_last
            dump(out/'registered_input_gate.json',dict(status='passed',whole_pack_aux_bitwise_equal=True,whole_pack_target_norm_bitwise_equal=True,registered_cpu_storages=len(registrations.buffers),utc=now()))
        assert hashlib.sha256(device['ids'].cpu().numpy().tobytes()).hexdigest()==meta['packed_ids_sha256']
        del pinned,host,payload
        acks.put(dict(event='consumed',arm=arm,pack=i+1,unix_time=time.time()))
        values=update(pair,target,basis,device,p['experiment_seed']+200299+i,1536,opt,check=i==0)
        end=time.perf_counter()
        record=dict(pack=i+1,arm=arm,measured=i>=p['warmup_packs'],total_update_seconds=end-begin,shared_queue_wait_seconds=received-begin,host_pin_seconds=pin_seconds,registered_shared_bf16=True,registered_cpu_storages=len(registrations.buffers),finished_unix_time=time.time(),input_tokens=meta['input_tokens'],examples=meta['examples'],packed_ids_sha256=meta['packed_ids_sha256'],**transfer,**values,utc=now())
        records.append(record)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        dump(out/'status.json',dict(status='training',pid=os.getpid(),latest=record,utc=now()))
        print(now(),'UPDATE',arm,i+1,'/',p['pack_count'],'seconds',round(record['total_update_seconds'],3),'queue_wait',round(record['shared_queue_wait_seconds'],3),'pin',round(pin_seconds,3),'body_fwd',round(values['body_forward_wall'],3),'logits_loss_fwd',round(values['head_forward_loss_wall'],3),'logits_bwd',round(values['head_backward_wall'],3),'body_bwd',round(values['body_backward_wall'],3),flush=True)
        acks.put(dict(event='updated',arm=arm,pack=i+1,finished_unix_time=record['finished_unix_time']))
        del device
    measured=[v for v in records if v['measured']]
    fields=['total_update_seconds','shared_queue_wait_seconds','host_pin_seconds','h2d_wall_seconds','body_forward_wall','projection_wall','head_forward_loss_wall','head_backward_wall','body_backward_wall','clip_wall','optimizer_wall']
    summary=dict(status='complete',arm=arm,updates=len(records),measured_updates=len(measured),head_chunk=1536,anchors_per_update=1536,pack_tokens=65536,means={k:statistics.mean(v[k] for v in measured) for k in fields},peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in records),loss='body raw CE with D-PACE + head repaired CE with D-PACE + OUTPUT' if arm=='joint' else 'raw CE with D-PACE',utc=now())
    dump(out/'summary.json',summary)
    save(out,pair,opt,len(records),p)
    dump(out/'status.json',dict(status='complete',summary=summary,utc=now()))
