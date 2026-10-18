"""CPU-only: real HTTP client against a mock endpoint, no model server/CUDA."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run


class Checks(unittest.TestCase):
    def test_sources_and_migration(self):
        root = HERE.parent
        for entry in json.loads((HERE / 'source_origin.json').read_text()):
            # original_sha256 remains the historical snapshot; explicitly
            # maintained files also record their current hash and reason.
            expected = entry.get('current_sha256', entry['original_sha256'])
            self.assertEqual(hashlib.sha256((root / entry['original']).read_bytes()).hexdigest(), expected)
        manifest = json.loads((HERE / 'data_migration_manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete_verified')
        for entry in manifest['directories']:
            old, new = Path(entry['old']), Path(entry['new'])
            self.assertTrue(old.is_symlink())
            for f in entry['files']:
                self.assertTrue((old / f['name']).samefile(new / f['name']))
                self.assertEqual((new / f['name']).stat().st_size, f['size'])

    def test_plan_and_no_implicit_gpu(self):
        with patch.object(sys, 'argv', ['run.py', '--dataset', 'openmathinstruct2']):
            args = run.parse_args()
        plan = run.make_plan(args)
        self.assertFalse(args.run)
        self.assertTrue(plan['output'].endswith('/openmath_qwen3_8b_onpolicy.jsonl'))
        self.assertEqual(plan['speculative_config']['num_speculative_tokens'], 7)
        self.assertEqual(plan['speculative_config']['draft_sample_method'], 'greedy')
        server = plan['server_argv']
        self.assertEqual(server[server.index('--max-num-seqs')+1], '50')
        self.assertNotIn('--enforce-eager', server)
        self.assertNotIn('--enable-lora', server)
        with patch.dict(os.environ, {'LD_PRELOAD': 'foreign.so', 'PYTHONPATH': '/our/patches', 'VLLM_USE_V2_MODEL_RUNNER': '1'}):
            env = run.environment(args)
        self.assertNotIn('LD_PRELOAD', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertNotIn('VLLM_USE_V2_MODEL_RUNNER', env)
        self.assertEqual(env['VLLM_ENABLE_CUDA_COMPATIBILITY'], '1')
        self.assertEqual(env['LD_LIBRARY_PATH'], f'{args.env_dir}/cuda-compat:{args.env_dir}/lib')
        result = subprocess.run([sys.executable, str(HERE/'run.py'), '--dataset', 'openmathinstruct2', '--run'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'requires explicit --gpu', result.stderr)

    def test_corrupt_tail_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'rows.jsonl'
            p.write_text('{"id":"a","input_ids":[1],"loss_mask":[1]}')
            with self.assertRaisesRegex(ValueError, 'Incomplete JSONL tail'):
                run.count_existing(p)

    def test_client_resume_concurrency_budget_and_error(self):
        state = {'active': 0, 'peak': 0, 'requests': [], 'count': 5}
        lock = threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    answer = {'count': state['count'], 'tokens': list(range(state['count']))}
                else:
                    with lock:
                        state['active'] += 1
                        state['peak'] = max(state['peak'], state['active'])
                        state['requests'].append(payload)
                    time.sleep(.08)
                    answer = {'prompt_token_ids': [1, 2, 3, 4, 5], 'choices': [
                        {'message': {'role': 'assistant', 'content': 'mock answer'},
                         'token_ids': [6, 7], 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}}
                    with lock:
                        state['active'] -= 1
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                raw = d/'raw';raw.mkdir()
                rows = [{'id': str(i), 'input': f'question {i}', 'output': 'old answer'} for i in range(4)]
                (raw/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [rows[0], rows[1], rows[1], *rows[2:]]))
                output = d/'answers.jsonl'
                existing = json.dumps({'id': '0_gen0', 'primary_id': '0', 'input_ids': [1,2], 'loss_mask': [0,1]})+'\n'
                output.write_text(existing)
                command = [sys.executable, str(HERE/'regenerate.py'), '--dataset', 'opencodeinstruct',
                           '--dataset-path', str(raw), '--model', str(run.MODELS/'Qwen3-8B'),
                           '--endpoint', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                           '--outfile', str(output), '--resume', '--limit', '3', '--concurrency', '3',
                           '--max-model-len', '12', '--max-tokens', '10', '--run-manifest', 'mock-manifest',
                           '--summary-file', str(d/'summary.json')]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1')
                result = subprocess.run(command, env=env, cwd=d, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(output.read_text().startswith(existing))
                actual = [json.loads(s) for s in output.read_text().splitlines()]
                self.assertEqual({r['primary_id'] for r in actual}, {'0','1','2','3'})
                self.assertEqual(len(actual), 4)
                self.assertGreaterEqual(state['peak'], 2)
                self.assertEqual(len(state['requests']), 3)
                for payload in state['requests']:
                    self.assertEqual(payload['max_completion_tokens'], 7)
                    self.assertTrue(payload['return_token_ids'])
                for row in actual[1:]:
                    self.assertEqual(row['loss_mask'], [0]*5+[1]*2)
                    self.assertEqual(row['input_ids'], [1,2,3,4,5,6,7])
                    self.assertEqual(row['metadata']['sampling_params']['max_completion_tokens'], 7)
                    self.assertEqual(row['metadata']['run_manifest'], 'mock-manifest')
                self.assertEqual(json.loads((d/'summary.json').read_text())['errors'], 0)
                # A prompt filling the entire context is an error, never a silently truncated row.
                state['count'] = 12
                failout = d/'failed.jsonl'
                command[command.index('--outfile')+1] = str(failout)
                command[command.index('--limit')+1] = '1'
                result = subprocess.run(command, env=env, cwd=d, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(failout.read_text(), '')
                self.assertEqual(len(state['requests']), 3)
                self.assertIn('context limit', (d/'failed.errors.jsonl').read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
