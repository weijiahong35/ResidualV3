"""Use completed matched benchmarks to confirm production and finish a rolling deployment."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import os
import sys

HERE=Path(__file__).resolve().parent
from tune_speculation import Session, read, production_window
from pool import write_json
from profiles import PROFILES
from speculation_policy import choose_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    parser.add_argument('--pilot-gpu',type=int,required=True)
    args=parser.parse_args()
    profile=PROFILES['nonthinking_ctx4096']
    audit=args.audit_dir.resolve();audit.mkdir(parents=True,exist_ok=True)
    lock=(profile.output_root/'speculation_tuning.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    results=[read(args.benchmark_dir/f'{label}-result.json') for label in ['k15','k7','k3','k5','k5-confirmation']]
    hashes={r['fixture_sha256'] for r in results}
    assert len(hashes)==1 and all(r['errors']==0 and r['processed']==r['expected']==128 for r in results)
    best,gain=choose_result(results)
    selected=best['num_speculative_tokens']
    assert selected==5 and gain>=.05,'Recorded trials must establish a repeatable gain'
    report={'state':'starting','controller_pid':os.getpid(),'pilot_gpu':args.pilot_gpu,
        'benchmark_dir':str(args.benchmark_dir.resolve()),'results':results,
        'selected_tokens':selected,'rolled_out':{},'benchmark_gain':gain}
    def save(**values):
        report.update(values,updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(audit/'status.json',report)
        print(json.dumps(values,ensure_ascii=False),flush=True)
    contract=profile.output_root/'generation_contract.json'
    checksum=hashlib.sha256(contract.read_bytes()).hexdigest()
    baseline=production_window(profile,args.pilot_gpu,90)
    save(stage='production_baseline_recorded',production_baseline=baseline)
    session=Session(profile,args.pilot_gpu,audit,save)
    try:
        session.begin();session.configure(selected,'Completed matched benchmark; confirm on a current production worker')
        restored=session.resume()
        confirmed=production_window(profile,args.pilot_gpu,180)
        production_gain=confirmed['output_tokens_per_second']/baseline['output_tokens_per_second']-1
        save(state='pilot_verified',stage='production_confirmed',production_confirmation=confirmed,
             production_gain=production_gain)
        if confirmed['preemptions'] or production_gain<.05:
            session=Session(profile,args.pilot_gpu,audit,save)
            session.begin();session.configure(15,'Production confirmation did not establish a five-percent gain');session.resume()
            save(state='complete',selected_tokens=15,stage='native_configuration_retained',rolled_out={})
            return
        policy={'num_speculative_tokens':selected,'native_num_speculative_tokens':15,
            'anchor_generation_contract_sha256':checksum,'override_scope':['num_speculative_tokens'],
            'target_sampling_unchanged':True,'model_weights_unchanged':True,
            'benchmark_audit_dir':str(args.benchmark_dir.resolve()),'rollout_audit_dir':str(audit),
            'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
        write_json(profile.output_root/'speculation_policy.json',policy)
        # Retire trial overrides on inactive cards, so a later idle-card claim
        # receives the measured global default instead of an abandoned trial.
        trial=profile.worker_dir(1)/'speculation.json'
        if trial.exists():write_json(trial,{**read(trial),'num_speculative_tokens':selected,'reason':'Completed trial; use the confirmed global default'})
        report['rolled_out'][str(args.pilot_gpu)]={'num_speculative_tokens':selected,
            'server_pid':restored['server_pid'],'client_pid':restored['client_pid'],
            'concurrency':restored['concurrency'],'production':confirmed}
        latest=read(profile.scheduler_dir/'status.json')
        for key in sorted(latest['workers'],key=lambda k:(int(k)==0,int(k))):
            index=int(key)
            if index==args.pilot_gpu:continue
            session=Session(profile,index,audit,save)
            session.begin();session.configure(selected,'Sequential rollout of the measured faster draft length')
            restored=session.resume()
            measured=production_window(profile,index,60)
            report['rolled_out'][key]={'num_speculative_tokens':selected,'server_pid':restored['server_pid'],
                'client_pid':restored['client_pid'],'concurrency':restored['concurrency'],'production':measured}
            save(state='rolling_out',rolled_out=report['rolled_out'])
        assert hashlib.sha256(contract.read_bytes()).hexdigest()==checksum
        save(state='complete',stage='all_current_workers_verified',generation_contract_unchanged=True,
            rolled_out=report['rolled_out'])
    except BaseException as error:
        save(state='failed',error=repr(error))
        if session.guard and session.old and __import__('hot_pool_extension').matching(session.old):
            try:
                session.configure(15,'Restore production after a failed rollout');session.resume()
                save(recovery='this_worker_restored; other workers remain running')
            except BaseException as recovery_error:
                save(recovery_error=repr(recovery_error))
        raise
    finally:
        session.unfreeze()


if __name__=='__main__':main()
