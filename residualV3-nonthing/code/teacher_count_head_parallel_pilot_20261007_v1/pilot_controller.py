"""Bounded throughput study; stop only its own workers and teacher service."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

CODE=Path(__file__).resolve().parent; RUNTIME=CODE.parent/'runtime_bundle'
sys.path.insert(0,str(RUNTIME))
from runtime.utils import dump,now,sha,stop,members


def inventory():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    return dict(gpu=gpu.splitlines(),processes=jobs.splitlines())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    output=r/'native_gpu4_v14';output.mkdir(exist_ok=False)
    p=json.loads((r/'protocol.json').read_text());service=r/'teacher_service_v14'
    # GPU7 now has another user's job; keep both allocator profiles on the empty card4.
    p=dict(p,learner_gpu=4,learner_uuid='GPU-b69b3699-6c78-06df-49e1-81ef44cf415b')
    dump(output/'learner_assignment.json',dict(gpu=4,uuid=p['learner_uuid'],utc=now()))
    hashes={str(x):sha(x) for x in CODE.iterdir() if x.suffix in ['.py','.c','.so']}
    dump(output/'SOURCE_FREEZE.json',hashes)
    gate=r/'allocator_recovery_v3/full_gate/summary.json'
    assert json.loads(gate.read_text())['status']=='passed'
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=p['learner_uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        PYTHONPATH=f'{CODE}:{RUNTIME}:{RUNTIME}/src:{RUNTIME}/hs_connectors/src',
        TRITON_CACHE_DIR=str(r/'triton_cache'),TORCHINDUCTOR_CACHE_DIR=str(r/'inductor_cache'))
    worker=None
    def interrupted(sig,frame):raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        existing=json.loads((r/'controlled_gpu4_v11/fixed/profiles/summary.json').read_text())
        assert existing['status']=='complete'
        results=list(existing['profiles'])
        for arm in ['native']:
            while True:
                info=inventory();free=int(subprocess.check_output(['nvidia-smi','--id',p['learner_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
                threshold=75000
                if free>=threshold:break
                dump(output/'status.json',dict(status='waiting_memory',arm=arm,minimum_free_mib=threshold,inventory=info,utc=now()));time.sleep(5)
            cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'controlled_worker.py'),
                 '--output',str(output/arm),'--source',p['source_run'],'--service',str(service),'--allocator',arm,'--gate',str(gate)]
            with (output/f'{arm}_training.log').open('x') as log:
                worker=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            dump(output/f'{arm}_launch.json',dict(pid=worker.pid,command=cmd,initial_inventory=info,source_sha256=hashes,utc=now()))
            dump(r/'ACTIVE_RUN.json',dict(controller_pid=os.getpid(),learner_pid=worker.pid,gpu=p['learner_gpu'],
                 log=str(output/f'{arm}_training.log'),output=str(output/arm),service=str(service),teacher_dp=2,teacher_tp=1,
                 teacher_kv_gib_each=48,workspace_gib=68 if arm=='fixed' else None,unique_request_cache_salt=True))
            while worker.poll() is None:
                dump(output/'status.json',dict(status='running',arm=arm,worker_pid=worker.pid,utc=now()))
                with (output/'gpu_monitor.jsonl').open('a') as stream:
                    endpoint_path=service/'endpoint.json'
                    ep=json.loads(endpoint_path.read_text()) if endpoint_path.exists() else None
                    teacher_members=members(ep['server_pid']) if ep else []
                    stream.write(json.dumps(dict(utc=now(),arm=arm,worker_pid=worker.pid,teacher_ready=ep is not None,teacher_members=teacher_members,inventory=inventory()))+'\n')
                time.sleep(5)
            result=worker.returncode;stop(worker);worker=None
            assert result==0,f'{arm} failed, exit={result}; see {output/arm}'
            data=json.loads((output/arm/'profiles/summary.json').read_text());assert data['status']=='complete'
            results.extend(data['profiles'])
        contamination=[]
        identities=json.loads((service/'endpoint.json').read_text())['identity'];target_uuids={x['uuid'] for x in identities}
        for line in (output/'gpu_monitor.jsonl').read_text().splitlines():
            d=json.loads(line)
            for row in d['inventory']['processes']:
                uuid,pid,mem=[x.strip() for x in row.split(',')]
                if (uuid==p['learner_uuid'] and int(pid)!=d['worker_pid']) or (uuid in target_uuids and int(pid) not in d['teacher_members']):
                    contamination.append(dict(utc=d['utc'],arm=d['arm'],uuid=uuid,pid=int(pid),memory_mib=int(mem)))
        with (output/'throughput.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        dump(output/'summary.json',dict(status='complete',profiles=results,contamination_observed=contamination,
             native_speed_claim=False,source_weights_untouched=True,unique_KV_cache_salt_each_request=True,utc=now()))
        text=['训练吞吐小实验：同卡4，同源第894步权重，同数据/anchor/目标；每组12次更新，丢弃前3包，测量9包。',
              '两张teacher：DP2 TP1；每卡48GiB KV、max_num_seqs64、max_num_batched_tokens65536。',
              '修复头1536整块反传；DFlash前向512分块；本体和target冻结。',
              '每个请求独立cache_salt，禁止跨请求KV复用；hidden在线算，读取即删，无持久hidden/logits。',
              'allocator requests 秒/包 wait DFlash head input_tokens/s']
        for d in results:text.append(f"{d['allocator']} {d['request_concurrency']} {d['mean_seconds']:.3f} {d['mean_hidden_wait_seconds']:.3f} {d['mean_draft_seconds']:.3f} {d['mean_head_seconds']:.3f} {d['input_tokens_per_second']:.1f}")
        text.append('Native参考：68GiB上限、67GiB缓存预热、expandable_segments=True；固定组68GiB常驻工作区。')
        text.append('这是训练工程观察，不是native推理吞吐结果；样本小且分组顺序不同。')
        text.append('观测到的额外GPU进程：'+str(len(contamination)))
        (output/'RESULTS_CN.txt').write_text('\n'.join(text)+'\n')
        dump(output/'status.json',dict(status='complete',results=str(output/'RESULTS_CN.txt'),utc=now()))
    except BaseException as exc:
        dump(output/'status.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),utc=now()));raise
    finally:
        if worker is not None:stop(worker)
        dump(service/'stop.json',dict(reason='bounded controlled throughput pilot ended',utc=now()))


if __name__=='__main__':main()
