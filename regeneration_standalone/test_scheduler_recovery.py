"""A GPU inventory failure must preserve independent generation processes."""
from contextlib import ExitStack
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parent/'open_perfectblend_regen'))
import scheduler
from pool import write_json


class EndObservation(Exception):pass


class SchedulerRecoveryChecks(unittest.TestCase):
    def exercise(self,intentional_stop):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            profile=SimpleNamespace(name='test',thinking='off',mode='nonthinking',
                max_model_len=4096,draft_model=Path('/pinned/draft'),
                generation_dir=root/'generated',output_root=root/'generated/nonthinking',
                scheduler_dir=root/'scheduler',worker_dir=lambda index:root/f'gpu{index}')
            worker={'supervisor_pid':123,'profile':'test','state':'generating',
                'gpu_index':1,'scheduling':'shared shard queue with nonblocking exclusive locks'}
            write_json(profile.scheduler_dir/'status.json',{'workers':{'1':worker},
                'gpu_inventory':[{'index':7,'idle':True,'compatible':True}]})
            write_json(profile.worker_dir(1)/'status.json',worker)
            progress={'all_processed':False,'splits':{'train':{'expected_conversations':4}},
                'total_shards':1,'shards':[{'complete':False}]}
            handlers={}
            kill=Mock();start=Mock()
            calls=0
            def inventory(scope):
                nonlocal calls
                calls+=1
                if intentional_stop:
                    handlers[signal.SIGTERM](signal.SIGTERM,None)
                if calls==1:
                    raise subprocess.TimeoutExpired(['nvidia-smi','-q','-x'],15)
                if calls==2:
                    return [{'index':1,'uuid':'GPU-test','idle':False,'compatible':True}]
                raise EndObservation
            with ExitStack() as stack:
                for context in [patch.object(scheduler,'PROFILES',{'test':profile}),
                    patch.object(scheduler,'add_profile_argument',lambda p:p.add_argument('--profile',default='test')),
                    patch.object(scheduler,'pool_progress',return_value=progress),
                    patch.object(scheduler,'gpu_inventory',side_effect=inventory),
                    patch.object(scheduler,'owned_process',side_effect=lambda pid,required:pid==123),
                    patch.object(scheduler.signal,'signal',side_effect=lambda sig,fn:handlers.update({sig:fn})),
                    patch.object(scheduler.os,'kill',kill),
                    patch.object(scheduler.subprocess,'Popen',start),
                    patch.object(scheduler.time,'sleep'),
                    patch.object(sys,'argv',['scheduler.py','--gpus','all'])]:stack.enter_context(context)
                with self.assertRaises(KeyboardInterrupt if intentional_stop else EndObservation):
                    scheduler.main()
            state=scheduler.read_json(profile.scheduler_dir/'status.json')
            return state,kill,start

    def test_query_timeout_retries_without_claiming_stale_idle_gpu_or_killing_workers(self):
        state,kill,start=self.exercise(False)
        kill.assert_not_called();start.assert_not_called()
        self.assertTrue(state['workers_retained'])
        self.assertIn('TimeoutExpired',state['gpu_inventory_error'])

    def test_explicit_scheduler_stop_still_releases_its_own_workers(self):
        _,kill,_=self.exercise(True)
        kill.assert_called_once_with(123,signal.SIGTERM)


if __name__=='__main__':unittest.main()
