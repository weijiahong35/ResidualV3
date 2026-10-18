"""Benchmark one owned GPU, then roll a verified draft length out one GPU at a time."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pool import allocate, write_json
from dispatch import gpu_inventory
from profiles import PROFILES, reservation_matches
from hot_pool_extension import identity, matching, send, stopped, wait_for
from speculation_policy import choose_result, execution_record
import pilot_gpu1 as pilot

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def metrics(port):
    with OPENER.open(f'http://127.0.0.1:{port}/metrics', timeout=8) as reply:
        body = reply.read().decode()
    values = {}
    for line in body.splitlines():
        if not line.startswith('vllm:'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[1])
    return values


def post(port, route):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/{route}', data=b'', method='POST')
    with OPENER.open(request, timeout=20) as reply:
        return reply.status


def detached(command, logfile, env=None):
    with Path(logfile).open('ab') as handle:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def stop_group(record, first=signal.SIGTERM):
    if record is None or not matching(record):
        return
    if os.getpgid(record['pid']) != record['pid']:
        raise RuntimeError('Recorded task is not its own process group')
    os.killpg(record['pid'], first)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.killpg(record['pid'], 0)
        except ProcessLookupError:
            return
        time.sleep(.5)
    # All group members were launched in this task's dedicated session.
    try:
        os.killpg(record['pid'], signal.SIGKILL)
    except ProcessLookupError:
        pass


def fixture(source, destination, count):
    groups = defaultdict(list)
    for line in Path(source).open():
        row = json.loads(line)
        groups[row['source']].append(row)
    quotas = allocate(count, {key:len(rows) for key,rows in groups.items()})
    if len(groups) != 8 or min(quotas.values()) < 1:
        raise ValueError('Benchmark must cover all eight source families')
    rng = random.Random(42)
    rows = [row for key in sorted(groups) for row in rng.sample(groups[key], quotas[key])]
    rng.shuffle(rows)
    destination.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
    return {'count':count,'source_counts':quotas,
            'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),
            'source_path':str(source)}


class Session:
    def __init__(self, profile, index, audit, save, reservation_pid=None):
        self.profile, self.index, self.audit, self.save = profile, index, audit, save
        self.directory = profile.worker_dir(index)
        self.old = None
        self.guard = None
        self.client = None
        self.new = None
        self.server = None
        self.before = None
        self.reservation_pid = reservation_pid
        self.shard_lock = None

    def begin(self):
        initial = read(self.directory/'status.json')
        if self.reservation_pid:
            self.before = initial
            self.uuid, self.port = initial['gpu_uuid'], initial['port']
            if not reservation_matches(self.reservation_pid,self.uuid):
                raise RuntimeError('Recovery reservation does not match this owned GPU')
            device=gpu_inventory(str(self.index))[0]
            if device['compute_pids'] != [self.reservation_pid]:
                raise RuntimeError('Recovery GPU has a compute process other than its reservation')
            self.old=identity(initial['supervisor_pid'],[str(HERE/'pilot_gpu1.py'),'--profile',self.profile.name])
            if self.old:
                raise RuntimeError('Reservation recovery requires the prior CPU supervisor to have exited')
            class AttachedGuard(pilot.AttachedClient):
                def terminate(guard):
                    if reservation_matches(guard.pid,self.uuid):
                        os.kill(guard.pid,signal.SIGTERM)
            self.guard=AttachedGuard(self.reservation_pid)
            shard_lock=Path(initial['output']).with_suffix('.lock').open('a')
            try:
                fcntl.flock(shard_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                shard_lock.close()
            else:
                self.shard_lock=shard_lock
            self.save(stage='adopted_retained_gpu_reservation',active_gpu=self.index,
                reservation_pid=self.guard.pid)
            return
        self.old = identity(initial['supervisor_pid'],
            [str(HERE/'pilot_gpu1.py'),'--gpu',str(self.index),'--profile',self.profile.name])
        if self.old is None or initial['state'] != 'generating':
            raise RuntimeError(f'GPU {self.index} is not generating stably')
        send(self.old, signal.SIGSTOP)
        wait_for(lambda:stopped(self.old), 10, 'CPU supervisor did not pause')
        self.before = read(self.directory/'status.json')
        self.uuid, self.port = self.before['gpu_uuid'], self.before['port']
        self.server = identity(self.before['server_pid'], ['vllm.entrypoints.cli.main',
            '--max-model-len',str(self.profile.max_model_len)])
        if self.server is None:
            raise RuntimeError('Owned API server exited before reservation')
        contract = read(self.profile.output_root/'generation_contract.json')
        execution_record(read(self.directory/'server.json'),contract,
                         self.profile.output_root/'speculation_policy.json')
        device = gpu_inventory(str(self.index))[0]
        if device['uuid'] != self.uuid or any(p['ownership'] != 'self' for p in device['compute_processes']):
            raise RuntimeError('GPU now contains another account; refusing handoff')
        for pid in device['compute_pids']:
            if os.getpgid(pid) != self.server['pid']:
                raise RuntimeError('GPU contains another job, even though its UID is ours')
        self.client = identity(self.before.get('client_pid'), [str(ROOT/'regenerate.py'),
            self.before['output'],f'http://127.0.0.1:{self.port}/v1/chat/completions']) if self.before.get('client_pid') else None
        command = [sys.executable,str(HERE/'gpu_handoff_reservation.py'),'--gpu-uuid',self.uuid,
            '--status-file',str(self.audit/f'gpu{self.index}-reservation.json'),'--lifetime-seconds','7200']
        env = pilot.backend.environment(SimpleNamespace(env_dir=Path(sys.executable).parent.parent, gpu=self.uuid))
        self.guard = detached(command, self.audit/f'gpu{self.index}-reservation.log', env)
        def holding():
            if self.guard.poll() is not None:
                raise RuntimeError('CUDA reservation failed; original model service stays alive')
            r = read(self.audit/f'gpu{self.index}-reservation.json')
            return r.get('state') == 'holding' and r.get('pid') == self.guard.pid
        wait_for(holding,90,'CUDA reservation did not initialize')
        assert reservation_matches(self.guard.pid,self.uuid)
        stop_group(self.client, signal.SIGINT)
        self.client = None
        wait_for(lambda:metrics(self.port).get('vllm:num_requests_running',0)==0,60,'Requests did not drain')
        job = {'output':self.before['output'],'source':str(HERE/'pool_500k_seed42/prompts/train'/
            f'shard-{self.before["active_shard"]:03d}.jsonl'),'shard':self.before['active_shard']}
        pilot.launcher.repair_output_tail(Path(job['output']))
        self.saved_count = pilot.launcher.completed_count(job)
        self.saved_skipped = pilot.launcher.skipped_count(job)
        shard_status=Path(job['output']).with_suffix('.status.json')
        write_json(shard_status,{**read(shard_status),'state':'partial',
            'completed_conversations':self.saved_count,'skipped_conversations':self.saved_skipped,
            'paused_for_speculation_tuning':str(self.audit)})
        write_json(self.directory/'status.json',{**self.before,'state':'tuning_speculation',
            'client_pid':None,'tuning_controller_pid':os.getpid(),
            'tuning_audit_dir':str(self.audit),'startup_reservation_pid':self.guard.pid,
            'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        self.save(stage='gpu_reserved_and_client_checkpointed',active_gpu=self.index,
            reservation_pid=self.guard.pid,saved_conversations=self.saved_count)

    def current_tokens(self):
        server = read(self.directory/'server.json')
        cmd = server['command']
        return json.loads(cmd[cmd.index('--speculative-config')+1])['num_speculative_tokens']

    def configure(self, tokens, reason, force_restart=False):
        write_json(self.directory/'speculation.json',{'num_speculative_tokens':tokens,
            'reason':reason,'audit_dir':str(self.audit),'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        if not force_restart and self.current_tokens() == tokens and self.server and matching(self.server):
            return
        self.save(stage='replace_one_model_server',active_gpu=self.index,num_speculative_tokens=tokens)
        stop_group(self.server)
        self.server = None
        wait_for(lambda:gpu_inventory(str(self.index))[0]['compute_pids']==[self.guard.pid],
            60,'Unexpected GPU compute task during reserved replacement')
        subprocess.run([sys.executable,str(HERE/'start_gpu1_server.py'),'--gpu',str(self.index),
            '--port',str(self.port),'--profile',self.profile.name,'--reservation-pid',str(self.guard.pid)],
            cwd=ROOT,check=True)
        info = read(self.directory/'server.json')
        def identified():
            self.server = identity(info['pid'],['vllm.entrypoints.cli.main'])
            return self.server is not None
        wait_for(identified,10,'Replacement server did not publish its command')
        def ready():
            if self.server is None or not matching(self.server):
                raise RuntimeError('Replacement vLLM server failed to start')
            try:
                metrics(self.port)
                return True
            except Exception:
                return False
        wait_for(ready,1500,'Replacement server did not become ready')
        info['state'] = 'ready'
        write_json(self.directory/'server.json',info)
        self.save(stage='model_server_ready',active_gpu=self.index,num_speculative_tokens=tokens,
                  server_pid=self.server['pid'])

    def resume(self):
        scheduler_state = read(self.profile.scheduler_dir/'status.json')
        scheduler = identity(scheduler_state['scheduler_pid'],[str(HERE/'scheduler.py'),
            '--profile',self.profile.name,'--gpus','all'])
        if scheduler is None:
            raise RuntimeError('Scheduler is not available for CPU supervisor handoff')
        send(scheduler,signal.SIGSTOP)
        try:
            wait_for(lambda:stopped(scheduler),10,'Scheduler did not pause for worker adoption')
            if self.old:
                send(self.old,signal.SIGKILL)
                wait_for(lambda:not matching(self.old),10,'Frozen CPU supervisor did not release its shard lock')
            if self.shard_lock:
                self.shard_lock.close()
                self.shard_lock=None
            command=[sys.executable,str(HERE/'pilot_gpu1.py'),'--gpu',str(self.index),
                '--gpu-uuid',self.uuid,'--port',str(self.port),'--profile',self.profile.name,
                '--reservation-pid',str(self.guard.pid)]
            self.new=detached(command,self.directory/'supervisor.log')
            write_json(self.directory/'supervisor_pid.json',{'pid':self.new.pid,'command':command,
                'gpu_index':self.index,'speculation_handoff':str(self.audit)})
            def published():
                if self.new.poll() is not None:
                    raise RuntimeError('New CPU supervisor exited during adoption')
                return read(self.directory/'status.json').get('supervisor_pid')==self.new.pid
            wait_for(published,45,'New CPU supervisor did not publish its identity')
            def generating():
                state=read(self.directory/'status.json')
                if self.new.poll() is not None or state.get('state') == 'failed':
                    raise RuntimeError(f'Replacement worker failed: {state.get("error")}')
                return (state.get('supervisor_pid')==self.new.pid and state.get('state')=='generating'
                    and state.get('client_progress',{}).get('ok',0)>=4)
            wait_for(generating,300,'Replacement did not save four conversations')
            state=read(self.directory/'status.json')
            if state['client_progress']['errors']:
                raise RuntimeError('Replacement client reported generation errors')
        finally:
            send(scheduler,signal.SIGCONT)
        self.guard.terminate()
        self.guard.wait(timeout=20)
        self.guard=None
        self.save(stage='gpu_generating_after_replacement',active_gpu=self.index,
            new_supervisor_pid=self.new.pid,server_pid=state['server_pid'],
            num_speculative_tokens=state['num_speculative_tokens'])
        return state

    def unfreeze(self):
        if self.old and matching(self.old):
            send(self.old,signal.SIGCONT)


def benchmark(session, audit, label, source, expected, client_template, save):
    # Every matched trial uses a freshly started server with an empty prefix
    # cache; the optional developer reset-prefix-cache API is not enabled here.
    output = audit/f'{label}.jsonl'
    summary = audit/f'{label}-summary.json'
    manifest = audit/f'{label}-run.json'
    command = list(client_template)
    for flag,value in [('--dataset-files',source),('--outfile',output),('--concurrency',40)]:
        command[command.index(flag)+1] = str(value)
    command += ['--limit',str(expected),'--summary-file',str(summary),'--run-manifest',str(manifest)]
    write_json(manifest,{'purpose':'isolated same-prompt serving benchmark; outside the training queue',
        'server':read(session.directory/'server.json'),'client_command':command,
        'fixture_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    env = pilot.backend.environment(SimpleNamespace(env_dir=Path(sys.executable).parent.parent,gpu=session.uuid),client=True)
    child = detached(command,audit/f'{label}-client.log',env)
    def identified():
        session.client=identity(child.pid,[str(ROOT/'regenerate.py'),str(output)])
        return session.client is not None
    wait_for(identified,10,'Benchmark client did not publish its command')
    started = time.monotonic()
    before = metrics(session.port)
    save(stage='benchmark_running',label=label,num_speculative_tokens=session.current_tokens(),
        active_gpu=session.index,benchmark_client_pid=child.pid)
    deadline = time.monotonic()+1200
    while child.poll() is None:
        if time.monotonic()>deadline:
            raise TimeoutError('Benchmark exceeded twenty minutes')
        if not matching(session.server):
            raise RuntimeError('Server exited during benchmark')
        time.sleep(2)
    session.client=None
    if child.returncode:
        raise RuntimeError(f'Benchmark {label} failed; inspect its client log')
    after=metrics(session.port)
    stats=read(summary)
    elapsed=stats['elapsed_seconds']
    drafts=after.get('vllm:spec_decode_num_drafts_total',0)-before.get('vllm:spec_decode_num_drafts_total',0)
    accepted=after.get('vllm:spec_decode_num_accepted_tokens_total',0)-before.get('vllm:spec_decode_num_accepted_tokens_total',0)
    result={'label':label,'num_speculative_tokens':session.current_tokens(),'concurrency':40,
        'expected':expected,'processed':stats['ok']+stats['skipped_context'],'errors':stats['errors'],
        'generated_conversations':stats['ok'],'terminal_skips':stats['skipped_context'],
        'requests':stats['requests'],'completion_tokens':stats['completion_tokens'],
        'elapsed_seconds':elapsed,'wall_seconds':time.monotonic()-started,
        'output_tokens_per_second':stats['completion_tokens']/elapsed,
        'conversations_per_hour':(stats['ok']+stats['skipped_context'])*3600/elapsed,
        'mean_acceptance_length':1+accepted/drafts if drafts else None,
        'preemptions':after.get('vllm:num_preemptions_total',0)-before.get('vllm:num_preemptions_total',0),
        'fixture_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'summary_path':str(summary)}
    if result['errors'] or result['processed'] != expected or result['preemptions']:
        raise RuntimeError('Benchmark has errors, missing conversations or GPU preemptions')
    write_json(audit/f'{label}-result.json',result)
    save(stage='benchmark_complete',latest_result=result)
    return result


def production_window(profile,index,seconds):
    state=read(profile.worker_dir(index)/'status.json')
    before=metrics(state['port'])
    start=time.monotonic()
    while time.monotonic()-start < seconds:
        current=read(profile.worker_dir(index)/'status.json')
        if current.get('state') != 'generating' or current.get('server_pid') != state['server_pid']:
            raise RuntimeError('Production service interrupted during throughput confirmation')
        if current.get('client_progress',{}).get('errors',0):
            raise RuntimeError('Production client has errors')
        time.sleep(2)
    after=metrics(state['port'])
    duration=time.monotonic()-start
    return {'seconds':duration,'output_tokens_per_second':
        (after.get('vllm:generation_tokens_total',0)-before.get('vllm:generation_tokens_total',0))/duration,
        'preemptions':after.get('vllm:num_preemptions_total',0)-before.get('vllm:num_preemptions_total',0)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=128)
    parser.add_argument('--rollout',action='store_true')
    parser.add_argument('--reservation-pid',type=int,help='Recover this task\'s retained reservation after a failed single-GPU handoff')
    parser.add_argument('--reuse-baseline',type=Path,help='Recorded original production baseline from the preceding attempt')
    args=parser.parse_args()
    profile=PROFILES['nonthinking_ctx4096']
    audit=args.audit_dir.resolve()
    audit.mkdir(parents=True,exist_ok=True)
    lock=(profile.output_root/'speculation_tuning.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    report={'state':'starting','controller_pid':os.getpid(),'pilot_gpu':args.gpu,
        'results':[],'rolled_out':{},'other_workers_continue':True}
    def save(**values):
        report.update(values,updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(audit/'status.json',report)
        print(json.dumps(values,ensure_ascii=False),flush=True)
    contract_path=profile.output_root/'generation_contract.json'
    contract_hash=hashlib.sha256(contract_path.read_bytes()).hexdigest()
    initial=read(profile.scheduler_dir/'status.json')
    write_json(audit/'before.json',initial)
    baseline=read(args.reuse_baseline)['production_baseline'] if args.reuse_baseline else production_window(profile,args.gpu,90)
    save(stage='production_baseline_recorded',production_baseline=baseline)
    session=Session(profile,args.gpu,audit,save,args.reservation_pid)
    try:
        session.begin()
        manifest=read(session.directory/f'train-{session.before["active_shard"]:03d}-run.json')
        source=audit/'matched_prompts.jsonl'
        metadata=fixture(manifest['job']['source'],source,args.samples)
        save(state='benchmarking',fixture=metadata)
        for tokens in [15,7,3,5]:
            session.configure(tokens,'single-GPU matched benchmark',force_restart=True)
            result=benchmark(session,audit,f'k{tokens}',source,args.samples,manifest['job']['client_argv'],save)
            report['results'].append(result)
            save(results=report['results'])
        best,gain=choose_result(report['results'])
        selected=best['num_speculative_tokens']
        if selected != 15:
            session.configure(selected,'repeat the fastest matched benchmark',force_restart=True)
            repeat=benchmark(session,audit,f'k{selected}-confirmation',source,args.samples,manifest['job']['client_argv'],save)
            report['results'].append(repeat)
            base_tps=report['results'][0]['output_tokens_per_second']
            if repeat['output_tokens_per_second'] < 1.05*base_tps:
                selected=15
        session.configure(selected,'confirmed single-GPU candidate')
        session.resume()
        window=production_window(profile,args.gpu,180)
        save(state='pilot_verified',selected_tokens=selected,benchmark_gain=gain,production_confirmation=window)
        if selected != 15 and (window['preemptions'] or
                window['output_tokens_per_second'] < baseline['output_tokens_per_second']*1.05):
            fallback=Session(profile,args.gpu,audit,save)
            session=fallback
            fallback.begin()
            fallback.configure(15,'Production gain was below five percent; keep native default')
            fallback.resume()
            selected=15
        write_json(profile.output_root/'speculation_policy.json',{
            'num_speculative_tokens':selected,'anchor_generation_contract_sha256':contract_hash,
            'native_num_speculative_tokens':15,'override_scope':['num_speculative_tokens'],
            'target_sampling_unchanged':True,'model_weights_unchanged':True,
            'benchmark_audit_dir':str(audit),'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
        report['rolled_out'][str(args.gpu)]={'num_speculative_tokens':selected,'production':window}
        if args.rollout and selected != 15:
            latest=read(profile.scheduler_dir/'status.json')
            for key in sorted(latest['workers'],key=lambda x:(int(x)==0,int(x))):
                index=int(key)
                if index == args.gpu:
                    continue
                current=read(profile.worker_dir(index)/'status.json')
                if current.get('state') != 'generating':
                    raise RuntimeError(f'GPU {index} is not stable for rolling deployment')
                session=Session(profile,index,audit,save)
                session.begin()
                session.configure(selected,'Rollout after matched benchmark and pilot production confirmation')
                restored=session.resume()
                measured=production_window(profile,index,60)
                report['rolled_out'][key]={'num_speculative_tokens':selected,
                    'server_pid':restored['server_pid'],'client_pid':restored['client_pid'],
                    'concurrency':restored['concurrency'],'production':measured}
                save(state='rolling_out',rolled_out=report['rolled_out'])
        assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==contract_hash
        save(state='complete',stage='all_requested_workers_verified',selected_tokens=selected,
            generation_contract_unchanged=True,rolled_out=report['rolled_out'])
    except BaseException as error:
        save(state='failed',error=repr(error))
        if session.client and matching(session.client):
            stop_group(session.client,signal.SIGINT)
            session.client=None
        # Restore this one GPU to production before abandoning any later rollout.
        if (session.old is None or matching(session.old)) and session.guard is not None:
            try:
                session.configure(15,'Recover the original stable serving configuration after tuning failure')
                session.resume()
                save(recovery='production_restored_on_native_draft_length')
            except BaseException as recovery_error:
                session.unfreeze()
                save(recovery_error=repr(recovery_error),
                    reservation_retained=bool(session.guard and session.guard.poll() is None))
        else:
            session.unfreeze()
        raise
    finally:
        session.unfreeze()


if __name__=='__main__':
    main()
