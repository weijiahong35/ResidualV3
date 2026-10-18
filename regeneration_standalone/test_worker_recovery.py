"""Exercise the actual supervisor's failed-client recovery without any GPU."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'open_perfectblend_regen'))
import pilot_gpu1 as pilot
from pool import write_json
from worker_recovery import refill_poll_delay, reserve_startup


class EndObservation(Exception):
    pass


class WorkerRecoveryChecks(unittest.TestCase):
    def run_supervisor(self, root, repeated_failure):
        worker = root / 'worker'
        worker.mkdir()
        write_json(worker / 'server.json', {'pid': 12345, 'gpu_uuid': 'GPU-test', 'port': 18081})
        write_json(worker / 'concurrency_results.json', {'selected_concurrency': 40, 'results': []})
        profile = SimpleNamespace(name='test', thinking='off', max_model_len=4096,
            draft_model=Path('/fake/draft'), generation_dir=root / 'generated', mode='nonthinking',
            worker_dir=lambda index: worker)
        jobs = []
        for n, target in enumerate([2, 1] if repeated_failure else [2]):
            output = root / f'generated/nonthinking/train/shard-{n:03d}.jsonl'
            jobs.append({'output': str(output), 'source': str(root / f'prompt-{n}.jsonl'),
                         'shard': n, 'split': 'train', 'target_conversations': target,
                         'full_shard_conversations': target,
                         'client_argv': ['mock-client', '--outfile', str(output)]})
        plan = {'generation_root': str(root / 'generated'), 'thinking': 'nonthinking',
                'contract': {'unchanged': True}, 'jobs': jobs}
        runtime = SimpleNamespace(model=Path('/fake/target'), draft_model=profile.draft_model, ready_timeout=1)
        committed = {job['output']: 0 for job in jobs}
        launches = []
        kill_server = Mock(side_effect=ProcessLookupError)

        def launch(argv, **kwargs):
            output = argv[argv.index('--outfile') + 1]
            attempt = sum(record['output'] == output for record in launches) + 1
            # The healthy API must remain resident when retrying or moving on.
            kill_server.assert_not_called()
            complete = output == jobs[-1]['output'] and (repeated_failure or attempt > 1)
            client = Mock(pid=20000 + len(launches), returncode=0 if complete else 1)

            def poll():
                if complete:
                    committed[output] = jobs[-1]['target_conversations']
                elif not repeated_failure:
                    committed[output] = 1
                return client.returncode

            client.poll.side_effect = poll
            launches.append({'output': output, 'limit': argv[argv.index('--limit') + 1]})
            return client

        with ExitStack() as stack:
            patches = [
                patch.object(pilot, 'HERE', root),
                patch.object(pilot, 'PROFILES', {'test': profile}),
                patch.object(pilot, 'add_profile_argument', lambda parser: parser.add_argument('--profile', default='test')),
                patch.object(pilot, 'load_pool_manifest', return_value=({'parameters': {'train_size': sum(j['target_conversations'] for j in jobs)}}, '', '')),
                patch.object(pilot, 'owned_process', return_value=True),
                patch.object(pilot, 'process_alive', return_value=True),
                patch.object(pilot, 'plans', return_value=[(runtime, plan)]),
                patch.object(pilot, 'metrics', return_value={}),
                patch.object(pilot, 'repair_output_tail'),
                patch.object(pilot.launcher, 'completed_count', side_effect=lambda job: committed[job['output']]),
                patch.object(pilot.launcher, 'skipped_count', return_value=0),
                patch.object(pilot.backend, 'environment', return_value={}),
                patch.object(pilot.backend, 'stop_group'),
                patch.object(pilot.subprocess, 'Popen', side_effect=launch),
                patch.object(pilot, 'export_training_paths'),
                patch.object(pilot.time, 'sleep'),
                patch.object(pilot.os, 'killpg', kill_server),
                patch.object(pilot.signal, 'signal'),
                patch.object(sys, 'argv', ['pilot_gpu1.py', '--gpu', '1', '--gpu-uuid', 'GPU-test']),
            ]
            for context in patches:
                stack.enter_context(context)
            if repeated_failure:
                stack.enter_context(patch.object(pilot, 'pool_progress', side_effect=EndObservation))
                with self.assertRaises(EndObservation):
                    pilot.main()
            else:
                stack.enter_context(patch.object(pilot, 'pool_progress', return_value={'all_processed': True}))
                pilot.main()
        return worker, jobs, launches, kill_server

    def test_one_client_error_resumes_only_missing_ids_on_same_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker, jobs, launches, kill_server = self.run_supervisor(Path(tmp), False)
            self.assertEqual([x['limit'] for x in launches], ['2', '1'])
            recovery = json.loads((worker / 'client_recovery.jsonl').read_text())
            self.assertEqual(recovery['server_pid'], 12345)
            self.assertTrue(recovery['server_retained'])
            status = json.loads(Path(jobs[0]['output']).with_suffix('.status.json').read_text())
            self.assertEqual(status['state'], 'complete')
            self.assertEqual(status['completed_conversations'], 2)
            # API shutdown occurs once, after the whole queue is complete.
            kill_server.assert_called_once_with(12345, pilot.signal.SIGTERM)

    def test_repeated_error_defers_shard_and_keeps_server_for_other_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker, jobs, launches, kill_server = self.run_supervisor(Path(tmp), True)
            self.assertEqual([x['output'] for x in launches], [jobs[0]['output']] * 3 + [jobs[1]['output']])
            failed = json.loads(Path(jobs[0]['output']).with_suffix('.status.json').read_text())
            self.assertEqual(failed['state'], 'partial')
            self.assertEqual(failed['retry_pending']['remaining_conversations'], 2)
            self.assertEqual(json.loads(Path(jobs[1]['output']).with_suffix('.status.json').read_text())['state'], 'complete')
            self.assertTrue(json.loads((worker / 'status.json').read_text())['server_retained'])
            kill_server.assert_not_called()

    def test_refill_is_fast_and_occupied_gpu_is_never_reserved(self):
        self.assertEqual(refill_poll_delay(20, 1, 2), 2)
        self.assertEqual(refill_poll_delay(20, 2, 2), 20)
        with patch('worker_recovery.subprocess.Popen') as start:
            with self.assertRaisesRegex(RuntimeError, 'idle compatible'):
                reserve_startup('GPU-test', Path('/tmp'), Mock(),
                    {'uuid': 'GPU-test', 'compatible': True, 'idle': False}, Mock())
            start.assert_not_called()


if __name__ == '__main__':
    unittest.main()
