"""Launch a bounded shared-card memory probe; clean up only its own server."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

CODE = Path(__file__).resolve().parent
RUNTIME = CODE.parent / 'runtime_bundle'
sys.path.insert(0, str(RUNTIME))
from runtime.utils import dump, sha, now, environment, healthy, stop, EVAL
from device import assignment, bind


def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True)
    tasks = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    occupied = {}
    for row in tasks.splitlines():
        uuid, pid = [value.strip() for value in row.split(',')]
        occupied.setdefault(uuid, []).append(int(pid))
    result = {}
    for row in rows.splitlines():
        index, uuid, total, free, util = [value.strip() for value in row.split(',')]
        result[int(index)] = dict(uuid=uuid, total_mib=int(total), free_mib=int(free), utilization=int(util),
                                  foreign_pids_at_selection=occupied.get(uuid, []))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source-run', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (out / 'protocol.json').exists(), 'No duplicate launch'
    devices = inventory()
    eligible = [i for i, d in devices.items() if d['free_mib'] >= 52000]
    assert len(eligible) >= 2, 'Two cards with sufficient actual free memory needed'
    ordered = sorted(eligible, key=lambda i: (bool(devices[i]['foreign_pids_at_selection']),
                                              devices[i]['utilization'], -devices[i]['free_mib']))
    learner, teacher = ordered[:2]
    hardware = dict(learner=assignment(learner), teacher=assignment(teacher), initial_inventory=devices)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    transfer = Path('/dev/shm') / f'residualV3_fusedloss_{os.getpid()}'
    transfer.mkdir(exist_ok=False)
    old = args.source_run / 'training_protocol.json'
    checkpoint = args.source_run / 'training/epoch_03_resume.pt'
    original = json.loads(old.read_text())
    protected = [old, checkpoint, Path(original['basis_path']),
                 args.source_run / 'training/epoch_03.safetensors',
                 RUNTIME / 'repair/model.py', RUNTIME / 'runtime/master_optimizer.py',
                 CODE.parent / 'packed20k_pipeline_v1/compute.py',
                 CODE.parent / 'packed20k_split_v3/train.py', *list(CODE.glob('*.py'))]
    # Keep 4GiB of existing free memory unclaimed; a capped OOM is an explicit scoped result.
    budget = min(int(devices[learner]['total_mib'] * 2**20 * .94),
                 devices[learner]['free_mib'] * 2**20 - 4 * 2**30)
    protocol = dict(original_training_protocol=str(old), checkpoint=str(checkpoint), port=port,
                    transfer_root=str(transfer), hardware=hardware, allocator_budget_bytes=budget,
                    protected_sha256={str(path):sha(path) for path in protected},
                    chunks=[64,256,1536], arms=['OUTPUT','CE_DPACE','CE_DPACE_OUTPUT'], temporary_updates_only=True,
                    teacher_and_learner_separate_gpus=True, shared_gpu_allowed=True,
                    foreign_processes_untouched=True, utc=now())
    dump(out / 'protocol.json', protocol)
    dump(out / 'launch.json', dict(controller_pid=os.getpid(), hardware=hardware, utc=now()))
    server = worker = None
    def interrupt(sig, frame):
        raise KeyboardInterrupt(str(sig))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        env = bind(environment(), hardware['teacher'])
        env.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                   TRITON_CACHE_DIR=str(out/'teacher_triton'), VLLM_USE_V2_MODEL_RUNNER='0')
        cmd = [str(EVAL/'bin/python'), '-B', str(RUNTIME/'scripts/launch_vllm.py'), original['target_model'],
               '--target-layer-ids','2','10','18','26','34','--hidden-states-path',str(transfer),'--',
               '--host','127.0.0.1','--port',str(port),'--tensor-parallel-size','1','--dtype','bfloat16',
               '--max-num-seqs','16','--max-model-len','4097','--max-num-batched-tokens','32768',
               '--gpu-memory-utilization','.43','--kv-cache-memory-bytes',str(12*2**30),
               '--enable-prefix-caching','--enforce-eager','--disable-uvicorn-access-log']
        with (out/'teacher_vllm.log').open('x') as log:
            server = subprocess.Popen(cmd, env=env, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL, start_new_session=True)
        dump(out/'teacher_launch.json', dict(pid=server.pid, command=cmd, utc=now()))
        deadline = time.monotonic()+900
        while not healthy(port):
            assert server.poll() is None, 'Teacher startup failed'
            assert time.monotonic()<deadline, 'Teacher startup timeout'
            dump(out/'controller_status.json', dict(status='running',stage='teacher_startup',pid=os.getpid(),
                                                    teacher_pid=server.pid,utc=now()))
            time.sleep(3)
        env = bind(os.environ.copy(), hardware['learner'])
        env.pop('LD_PRELOAD',None)
        env.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
                   PYTHONPATH=f'{RUNTIME}:{RUNTIME}/src:{RUNTIME}/hs_connectors/src',
                   TORCHINDUCTOR_CACHE_DIR=str(out/'training_inductor'),TRITON_CACHE_DIR=str(out/'training_triton'))
        cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'probe.py'),'--output',str(out)]
        with (out/'probe.log').open('x') as log:
            worker=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL,start_new_session=True)
        dump(out/'worker_launch.json',dict(pid=worker.pid,command=cmd,utc=now()))
        while worker.poll() is None:
            dump(out/'controller_status.json',dict(status='running',stage='fused_loss_probe',pid=os.getpid(),
                                                   worker_pid=worker.pid,utc=now()))
            with (out/'gpu_monitor.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=now(),devices=inventory()))+'\n')
            time.sleep(3)
        assert worker.returncode==0, f'Probe worker failed: {worker.returncode}'
        assert json.loads((out/'summary.json').read_text())['status']=='complete'
        summary=json.loads((out/'summary.json').read_text())
        assert summary['fused_gate_passed'] and summary['full1536_passed'], '1536 full block is required before training'
        # OUTPUT baseline is the already completed 20k packed run: same seed,
        # data/basis, anchor plan, optimizer/schedule and 894 updates. Preserve it.
        dump(out/'comparison_plan.json',dict(
            arms=['OUTPUT','CE_DPACE','CE_DPACE_OUTPUT'],
            output_baseline=str(args.source_run), output_baseline_retrained=False,
            new_train_arms=['CE_DPACE','CE_DPACE_OUTPUT'], train_count=20000,epochs=3,
            max_anchors_per_pack=1536,pack_tokens=65536, head_chunk_anchors=1536,
            head_mode=summary['selected_mode'], teacher_forced_only_parallel=True,
            native_predecessors_recursive=True, ce_label='current teacher full-vocabulary argmax',
            dpace_alpha=.5, combined_coefficients=dict(CE_DPACE=1.0,OUTPUT=1.0),
            limitation='new groups use validated fused loss/full-head block; old OUTPUT had chunk64; finite-precision accumulation can differ',utc=now()))
        execution_path=out/'execution_protocol.json'
        source_sha={str(path):sha(path) for path in CODE.glob('*.py')}
        for arm in ['CE_DPACE','CE_DPACE_OUTPUT']:
            armout=out/'training_arms'/arm
            armout.mkdir(parents=True,exist_ok=True)
            ap=original.copy()
            ap.update(output=str(armout),port=port,transfer_root=str(transfer),
                      loss_arm=arm,head_mode=summary['selected_mode'],head_chunk_anchors=1536,
                      allocator_budget_bytes=budget)
            protocol_path=armout/'training_protocol.json'
            dump(protocol_path,ap)
            dump(execution_path,dict(source_sha256=source_sha,training_protocol_sha256=sha(protocol_path)))
            env.update(PACKED_OPTIMIZATION_PROTOCOL=str(execution_path),PACKED_RECOVERY_ATTEMPT=str(armout))
            cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'train.py'),'--protocol',str(protocol_path)]
            with (armout/'training.log').open('x') as log:
                worker=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL,start_new_session=True)
            dump(armout/'launch.json',dict(pid=worker.pid,command=cmd,utc=now()))
            while worker.poll() is None:
                dump(out/'controller_status.json',dict(status='training',stage=arm,pid=os.getpid(),worker_pid=worker.pid,utc=now()))
                with (out/'gpu_monitor.jsonl').open('a') as stream: stream.write(json.dumps(dict(utc=now(),devices=inventory()))+'\n')
                time.sleep(5)
            assert worker.returncode==0, f'{arm} failed: {worker.returncode}'
        stop(server)
        server=None
        roots={'OUTPUT':args.source_run,'CE_DPACE':out/'training_arms/CE_DPACE','CE_DPACE_OUTPUT':out/'training_arms/CE_DPACE_OUTPUT'}
        old_native=json.loads((args.source_run/'native_protocol.json').read_text())
        checkpoints={}
        for arm,root in roots.items():
            weight=root/'training/epoch_03.safetensors'
            assert json.loads((root/'training/summary.json').read_text())['status']=='complete'
            checkpoints[arm]=dict(path=str(weight),sha256=sha(weight),head_config_path=original['head_config_path'],head_config_sha256=sha(original['head_config_path']))
        checkpoints['RAW']=dict(checkpoints['OUTPUT'])
        native=dict(old_native,native_arms=['RAW','OUTPUT','CE_DPACE','CE_DPACE_OUTPUT'],native_checkpoints=checkpoints,
                    native_configs={arm:{} for arm in checkpoints},architecture_by_arm={arm:original['architecture'] for arm in checkpoints},
                    source_files={**old_native['source_files'],**source_sha},
                    training_completions={arm:sha(root/'training/summary.json') for arm,root in roots.items()},
                    comparison='three losses, same 20k x3 packed anchor/data/initialization budget; reused completed OUTPUT, new fused/full-block accumulation',
                    asset_sha256={**old_native['asset_sha256'],**{row['path']:row['sha256'] for row in checkpoints.values()}})
        dump(out/'native_protocol.json',native)
        while True:
            devices=inventory()
            candidates=[i for i,d in devices.items() if d['free_mib']>=61440]
            if candidates: break
            dump(out/'controller_status.json',dict(status='waiting_evaluation_memory',minimum_free_mib=61440,utc=now()))
            time.sleep(5)
        evaluation_gpu=min(candidates,key=lambda i:(bool(devices[i]['foreign_pids_at_selection']),devices[i]['utilization'],-devices[i]['free_mib']))
        for concurrency,prompts in [(1,200),(40,2000)]:
            for smoke in [True,False]:
                phase=f'C{concurrency}_'+('smoke' if smoke else 'formal')
                destination=out/f'C{concurrency}'/('formal_smoke' if smoke else 'formal')
                cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'evaluate.py'),
                     '--training',str(out),'--gpu',str(evaluation_gpu),'--concurrency',str(concurrency),
                     '--output',str(destination),'--tokens',str(32 if smoke else 2096),
                     '--prompts',str(concurrency if smoke else prompts),'--rounds','1']
                if smoke: cmd.append('--smoke')
                with (out/f'single_card_{phase}.log').open('x') as log:
                    worker=subprocess.Popen(cmd,env=os.environ.copy(),cwd=CODE,stdout=log,stderr=subprocess.STDOUT,
                                            stdin=subprocess.DEVNULL,start_new_session=True)
                while worker.poll() is None:
                    dump(out/'controller_status.json',dict(status='evaluating',stage=phase,pid=os.getpid(),worker_pid=worker.pid,gpu=evaluation_gpu,utc=now()))
                    time.sleep(5)
                assert worker.returncode==0,f'{phase} failed'
        cmd=['/data/weijiahong/miniconda3/envs/dflash/bin/python','-B',str(CODE/'report.py'),'--output',str(out)]
        subprocess.run(cmd,check=True,cwd=CODE)
        dump(out/'controller_status.json',dict(status='complete',stage='complete',results=str(out/'RESULTS_CN.txt'),utc=now()))
    except BaseException as error:
        dump(out/'controller_status.json',dict(status='failed',error=repr(error),traceback=traceback.format_exc(),utc=now()))
        raise
    finally:
        for process in (worker,server):
            if process is not None:
                stop(process)
        files=list(transfer.iterdir())
        # Owned transient hidden files only; never remove any training evidence or weights.
        for path in files:
            if path.is_file():path.unlink()
        transfer.rmdir()
        dump(out/'cleanup.json',dict(owned_server_stopped=True,transient_hidden_files_removed=len(files),
                                    source_weights_retained=True,foreign_processes_untouched=True,utc=now()))


if __name__=='__main__':
    main()
