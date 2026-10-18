"""Bounded native correctness checks of a saved training snapshot, never formal timing."""
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);ap.add_argument('--gpu',type=int,default=5)
    a=ap.parse_args();training=a.training.resolve();out=training/'native_preflight_gpu5';out.mkdir(exist_ok=False)
    p=json.loads((training/'preregistration.json').read_text());sys.path.insert(0,p['runtime'])
    from runtime.utils import dump,sha,now,resources,stop
    from device import assignment
    import torch
    from safetensors.torch import save_file
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity=assignment(a.gpu);free,ids=resources(a.gpu)
    assert not ids and free>=75000, 'Only an empty whole card is eligible'
    state=torch.load(training/'training/recovery_checkpoints/resume_latest.pt',map_location='cpu',weights_only=True)
    assert 0<state['step']<60000 and state['protocol_sha256']==sha(training/'preregistration.json')
    checkpoint=out/'snapshot_head.safetensors';save_file({k:v.contiguous() for k,v in state['head'].items()},str(checkpoint))
    snapshot=dict(step=state['step'],checkpoint_sha256=sha(checkpoint),snapshot_only=True,final_training_checkpoint=False,utc=now())
    dump(out/'snapshot.json',snapshot);del state
    basis=training/'basis/repair_basis_512.safetensors'
    record=dict(path=str(checkpoint),sha256=sha(checkpoint),head_config_path=p['head_config_path'],head_config_sha256=sha(p['head_config_path']))
    p.update(gpu=a.gpu,uuid=identity['uuid'],basis_path=str(basis),basis_sha256=sha(basis),native_checkpoints={name:dict(record) for name in p['native_arms']},native_configs={name:{} for name in p['native_arms']},architecture_by_arm={name:p['architecture'] for name in p['native_arms']},training_completions={},training_snapshot=snapshot)
    original=Path(__file__).with_name('evaluate.py');fork=Path(__file__).with_name('native_preflight_evaluate.py')
    source=original.read_text()
    source=source.replace('def main():\n','def main():\n',1)
    source=source.replace('identity=assignment(a.gpu)','assert a.smoke, "Preflight permits correctness smoke only"\n    identity=assignment(a.gpu)',1)
    source=source.replace('training_may_overlap_test=False,all_training_complete_before_test=True','training_may_overlap_test=True,all_training_complete_before_test=False')
    source=source.replace("fixed final C1/C40 evaluation", "bounded native preflight of saved training snapshot")
    source=source.replace("same-card nonthinking RAW + scratch no-soft3; 20k x3; native block16=anchor+15 proposals; max2096; one forward round", "native snapshot preflight only; training ongoing on another GPU; block16=anchor+15 proposals; no speed evidence")
    source=source.replace("protocol.update(warmup_unique_prompts=", "protocol.update(training_snapshot=p['training_snapshot'],final_weights=False)\n    protocol.update(warmup_unique_prompts=")
    assert not fork.exists();fork.write_text(source)
    p['source_files']=dict(p['source_files'],**{str(Path(__file__).resolve()):sha(__file__),str(fork):sha(fork)})
    dump(out/'native_protocol.json',p)
    active=None
    def interrupt(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        dump(out/'launch.json',dict(controller_pid=os.getpid(),gpu_identity=identity,initial_host_pids=ids,initial_free_mib=free,snapshot=snapshot,utc=now()))
        for concurrency in [1,40]:
            cmd=[sys.executable,'-B',str(fork),'--training',str(out),'--gpu',str(a.gpu),'--concurrency',str(concurrency),'--output',str(out/f'C{concurrency}/formal_smoke'),'--tokens','32','--prompts',str(concurrency),'--smoke']
            with (out/f'C{concurrency}_smoke.log').open('x') as log:
                active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            dump(out/'status.json',dict(status='running',stage=f'C{concurrency}_native_correctness',child_pid=active.pid,controller_pid=os.getpid(),snapshot=snapshot,utc=now()))
            while active.poll() is None:time.sleep(3)
            assert active.returncode==0,f'C{concurrency} preflight failed; evidence preserved'
        dump(out/'status.json',dict(status='complete',snapshot=snapshot,diagnostic_only=True,speed_evidence=False,utc=now()))
    except BaseException as exc:
        dump(out/'status.json',dict(status='failed',error=repr(exc),snapshot=snapshot,diagnostic_only=True,utc=now()));raise
    finally:
        if active and active.poll() is None:stop(active)

if __name__=='__main__':main()
