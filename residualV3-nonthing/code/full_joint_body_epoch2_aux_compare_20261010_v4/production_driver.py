"""Real updates, paired checkpoint barriers and exact complete-state restore."""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from activation_probe import reserve_workspace
from full_checkpoint import atomic, final_weights, restore, save
from host_pipeline import upload
from joint_runner import update
from runtime.utils import now
from training_progress import lr_scale, resume_spec


def run_worker(root,arm,p,target,pair,opt,basis,in_queue,acks):
    from registered_storage import RegisteredStoragePool
    out=root/arm;job=Path(p['job_root']);proof=json.loads((job/'identity_proof.json').read_text())
    initial=json.loads((root/'start_cursor.json').read_text());cursor=initial['cursor'];completed=0
    checkpoint, restore_proof, expected_cursor = resume_spec(initial,p,proof)
    result=restore(checkpoint['artifacts'][arm],opt,restore_proof)
    assert result['cursor']==expected_cursor
    assert result['global_step']==p['global_step_offset']+cursor
    atomic(out/'restore_gate.json',dict(**result,resume_kind=initial['resume_kind'],
        local_cursor=cursor,source=str(checkpoint['artifacts'][arm]['path'])))
    print(now(),'RESTORED',arm,'epoch',p['training_epoch'],'local_cursor',cursor,
          'global_step',result['global_step'],'weights, Muon/Adam, master and RNG matched',flush=True)
    if arm=='joint':
        initialized=opt.auxiliary.activate()
        if initialized or initial['resume_kind']=='augmentation':opt.auxiliary_gate_complete=False
        print(now(),'AUX_COMPARISON_READY','Q45_MLP 1538 parameters; Q45_REPAIR_LATENT_MLP 9730 parameters',
              'fresh_at_cursor',cursor if initialized else None,flush=True)
    reserve_workspace(out/'allocator_pool.json',p['allocator_reserve_gib'])
    with RegisteredStoragePool() as registrations:
        while True:
            begin=time.perf_counter();payload=in_queue.get(timeout=900);received=time.perf_counter()
            if payload.get('event')=='checkpoint':
                assert payload['cursor']==cursor
                descriptor=save(job,arm,payload['slot'],cursor,opt,proof)
                if completed==1:
                    atomic(out/'full_state_roundtrip_gate.json',restore(descriptor,opt,proof))
                atomic(out/'last_checkpoint.json',descriptor)
                print(now(),'CHECKPOINT_SAVED',arm,'cursor',cursor,'GiB',round(descriptor['logical_bytes']/2**30,3),flush=True)
                acks.put(dict(event='checkpoint_saved',arm=arm,cursor=cursor,descriptor=descriptor))
                command=in_queue.get(timeout=900)
                assert command['event']=='checkpoint_committed' and command['cursor']==cursor
                if command['action']=='complete':
                    destination=final_weights(job,arm,opt)
                    atomic(out/'final_weight.json',dict(path=destination,cursor=cursor));break
                if command['action']=='pause':break
                continue
            assert payload['event']=='pack' and payload['pack']==cursor+1
            meta=payload['metadata'];fields=['ids','docs','positions','eligible','aux','final']
            assert all(payload[k].is_shared() for k in fields)
            t=time.perf_counter();pinned={k:registrations.pin(payload[k]) for k in fields};pin_seconds=time.perf_counter()-t
            host=SimpleNamespace(**pinned,offset=meta['input_tokens']);device,transfer=upload(host,target)
            if completed==0:
                assert torch.equal(device['aux'].cpu(),host.aux)
                with torch.no_grad():reference_last=target.model.norm(host.final.to('cuda'))
                assert torch.equal(device['last'],reference_last);del reference_last
                atomic(out/'registered_input_gate.json',dict(status='passed',whole_pack_aux_bitwise_equal=True,whole_pack_target_norm_bitwise_equal=True))
            assert hashlib.sha256(device['ids'].cpu().numpy().tobytes()).hexdigest()==meta['packed_ids_sha256']
            del pinned,host,payload
            acks.put(dict(event='consumed',arm=arm,pack=cursor+1))
            opt.lr_scale=lr_scale(cursor+1,p['pack_count'],p['lr_warmup_fraction'])
            recheck=job/'auxiliary_recheck.request.json'
            if arm=='joint' and recheck.exists():
                recheck.replace(job/'auxiliary_recheck.consumed.json')
                opt.auxiliary_gate_complete=False
            if arm=='joint' and not getattr(opt,'auxiliary_gate_complete',False):
                from auxiliary_live_gate import run as run_auxiliary_gate
                gate=run_auxiliary_gate(out,p,pair,target,opt,basis,device,
                    p['experiment_seed']+200299+p['global_step_offset']+cursor)
                atomic(job/'gates/auxiliary_live_gate.json',gate)
                print(now(),'AUXILIARY_Q45_GATE',gate['status'],'overhead',
                      gate['compute_overhead_fraction'],'main_gradients_bitwise',
                      gate['gradient_gate']['main_parameter_gradients_bitwise_equal'],flush=True)
            values=update(pair,target,basis,device,p['experiment_seed']+200299+p['global_step_offset']+cursor,1536,opt,check=completed==0)
            cursor+=1;completed+=1;end=time.perf_counter()
            record=dict(pack=cursor,epoch=p['training_epoch'],global_step=p['global_step_offset']+cursor,
                arm=arm,total_update_seconds=end-begin,shared_queue_wait_seconds=received-begin,
                host_pin_seconds=pin_seconds,finished_unix_time=time.time(),input_tokens=meta['input_tokens'],examples=meta['examples'],
                packed_ids_sha256=meta['packed_ids_sha256'],lr_scale=opt.lr_scale,real_parameter_update=True,**transfer,**values,utc=now())
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic(out/'status.json',dict(status='training',pid=os.getpid(),epoch=p['training_epoch'],
                cursor=cursor,global_step=p['global_step_offset']+cursor,total_packs=p['pack_count'],latest=record,utc=now()))
            if completed==1:
                atomic(out/'first_live_gradients.json',dict(status='passed',body_and_head_trainable=arm=='joint',target_frozen=True,gradients=values['gradients'],cursor=cursor))
            print(now(),'UPDATE',arm,'epoch',p['training_epoch'],cursor,'/',p['pack_count'],
                'global_step',p['global_step_offset']+cursor,'loss',round(values['loss'],6),
                'seconds',round(record['total_update_seconds'],3),
                'wait',round(record['shared_queue_wait_seconds'],3),'body_fwd',round(values['body_forward_wall'],3),
                'head_loss_fwd',round(values['head_forward_loss_wall'],3),'head_bwd',round(values['head_backward_wall'],3),
                'body_bwd',round(values['body_backward_wall'],3),'lr_scale',round(opt.lr_scale,6),flush=True)
            if arm=='joint' and 'auxiliary_loss' in values:
                for name,branch in values['auxiliary_branches'].items():
                    tag='AUX_Q45_MLP' if name=='Q45_MLP' else 'AUX_Q45_LATENT_MLP'
                    print(now(),tag,'steps',values['auxiliary_steps'],'loss',round(branch['loss'],6),
                          'T0_loss',round(branch['T0_loss'],6),'T1_loss',round(branch['T1_loss'],6),
                          'shared_feature_seconds',round(values['auxiliary_feature_wall'],4),
                          'both_predictors_seconds',round(values['auxiliary_training_wall'],4),flush=True)
            acks.put(dict(event='updated',arm=arm,pack=cursor,finished_unix_time=record['finished_unix_time']));del device
    atomic(out/'summary.json',dict(status='complete' if cursor==p['pack_count'] else 'paused',arm=arm,cursor=cursor,
        updates_in_attempt=completed,training_rows=p['training_rows'],epochs=1,training_epoch=p['training_epoch'],
        global_step=p['global_step_offset']+cursor,pack_tokens=65536,max_anchors=1536,head_chunk=1536,utc=now()))
