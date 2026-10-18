"""Retry only a failed resource-only C40 preflight; never restart training/C1."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True)
    training=ap.parse_args().training.resolve();parent=training/'native_preflight_gpu5'
    p=json.loads((parent/'native_protocol.json').read_text());sys.path.insert(0,p['runtime'])
    from runtime.utils import dump,sha,now,resources,stop
    original=json.loads((parent/'status.json').read_text());assert original['status']=='failed'
    c1=json.loads((parent/'C1/formal_smoke/summary.json').read_text());assert c1['status']=='complete'
    evidence=(parent/'C40/formal_smoke/round_1/RAW/vllm.log').read_text()
    assert 'CUDA out of memory' in evidence, 'Retry only genuine OOM, preserve non-resource failures'
    out=parent/'C40_resource_recovery_v1';out.mkdir(exist_ok=False)
    p['source_files']=dict(p['source_files'],**{str(Path(__file__).resolve()):sha(__file__)})
    dump(out/'native_protocol.json',p)
    cmd=[sys.executable,'-B',str(Path(__file__).with_name('native_preflight_evaluate.py')),'--training',str(out),'--gpu',str(p['gpu']),'--concurrency','40','--output',str(out/'C40/formal_smoke'),'--tokens','32','--prompts','40','--smoke']
    active=None
    def interrupt(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        while True:
            free,ids=resources(p['gpu'])
            if not ids and free>75000:break
            dump(out/'status.json',dict(status='waiting_empty_gpu',host_pids=ids,free_mib=free,utc=now()));time.sleep(10)
        with (out/'C40_smoke.log').open('x') as log:
            active=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(out/'launch.json',dict(controller_pid=os.getpid(),child_pid=active.pid,initial_host_pids=ids,initial_free_mib=free,utc=now()))
        dump(out/'status.json',dict(status='running',stage='C40_correctness_recovery',utc=now()))
        while active.poll() is None:time.sleep(3)
        assert active.returncode==0, 'C40 recovery failed; preserve all attempts'
        c40=json.loads((out/'C40/formal_smoke/summary.json').read_text());assert c40['status']=='complete'
        dump(out/'status.json',dict(status='complete',diagnostic_only=True,speed_evidence=False,utc=now()))
        dump(training/'native_preflight_summary.json',dict(status='complete',training_snapshot=p['training_snapshot'],C1_summary=str(parent/'C1/formal_smoke/summary.json'),C40_summary=str(out/'C40/formal_smoke/summary.json'),initial_C40_resource_failure_preserved=True,foreign_processes_untouched=True,training_unchanged=True,speed_evidence=False,utc=now()))
    except BaseException as exc:
        dump(out/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise
    finally:
        if active and active.poll() is None:stop(active)

if __name__=='__main__':main()
