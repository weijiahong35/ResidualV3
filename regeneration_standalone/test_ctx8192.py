"""Exercise real client budgeting, overflow recovery and disjoint GPU plans on CPU."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / '.client_deps'))
sys.path.insert(0, str(HERE / 'open_perfectblend_regen'))
os.environ.update(CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1')
from regenlib.checkpoint import load_skipped, skipped_path
import pilot_gpu1


class ContextChecks(unittest.TestCase):
    def test_shared_gpu_contracts(self):
        one = pilot_gpu1.plans(1, profile_name='thinking_ctx8192')
        three = pilot_gpu1.plans(3, profile_name='thinking_ctx8192')
        a = {job['shard'] for job in one[0][1]['jobs']}
        b = {job['shard'] for job in three[0][1]['jobs']}
        self.assertEqual(a, set(range(50)))
        self.assertEqual(b, set(range(50)))
        self.assertEqual([plan['split'] for _, plan in one], ['train'])
        self.assertEqual([plan['split'] for _, plan in three], ['train'])
        self.assertEqual(one[0][1]['contract'], three[0][1]['contract'])

    def test_exact_budget_multiturn_overflow_and_skip_resume(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                first = payload['messages'][0]['content']
                count = 8190 if first == 'SKIP' else 8192 if len(payload['messages']) > 1 else 2000
                if self.path == '/tokenize':
                    answer = {'count': count}
                else:
                    requests.append(payload)
                    answer = {'prompt_token_ids': [1] * count, 'choices': [
                        {'message': {'role': 'assistant', 'content': 'whole generated answer'},
                         'token_ids': [2] * 16, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': count, 'completion_tokens': 16}}
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
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                prompts = directory / 'prompts.jsonl'
                prompts.write_text(''.join(json.dumps({'id': first, 'source': 'mock/source',
                    'conversations': [{'role': 'user', 'content': first}] +
                    ([{'role': 'user', 'content': 'follow-up'}] if first == 'MULTI' else [])}) + '\n'
                    for first in ['FIT', 'MULTI', 'SKIP']))
                output = directory / 'answers.jsonl'
                command = [sys.executable, str(HERE / 'regenerate.py'), '--dataset', 'open-perfectblend',
                    '--dataset-files', str(prompts), '--model', str(pilot_gpu1.backend.MODELS / 'Qwen3-8B'),
                    '--endpoint', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    '--outfile', str(output), '--resume', '--concurrency', '3',
                    '--max-model-len', '8192', '--max-tokens', '8192', '--minimum-valid-tokens', '8',
                    '--skip-context-overflow', '--checkpoint-dir', str(directory / 'checkpoints'),
                    '--summary-file', str(directory / 'summary.json')]
                def run_client():
                    result = subprocess.run(command, cwd=directory, env=dict(os.environ, PYTHONNOUSERSITE='1'),
                                            capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                run_client()
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual({row['primary_id'] for row in rows}, {'FIT', 'MULTI'})
                self.assertEqual(len(requests), 2)
                for payload in requests:
                    self.assertEqual(payload['max_completion_tokens'], 6192)
                for row in rows:
                    self.assertEqual(len(row['input_ids']), 2016)
                    self.assertEqual(sum(row['loss_mask']), 16)
                    self.assertTrue(row['metadata']['conversation_complete'])
                multi = next(row for row in rows if row['primary_id'] == 'MULTI')
                self.assertIn('context limit', multi['metadata']['context_truncation_reason'])
                self.assertEqual(load_skipped(output), {'SKIP'})
                self.assertEqual(json.loads((directory / 'summary.json').read_text())['skipped_context'], 1)
                before = output.read_bytes()
                run_client()
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(len(requests), 2)
                with skipped_path(output).open('ab') as handle:
                    handle.write(b'{"primary_id":"torn')
                self.assertEqual(load_skipped(output), {'SKIP'})
                self.assertEqual(len(list(directory.glob('answers.skipped.jsonl.interrupted-tail-*.bin'))), 1)
                run_client()
                self.assertEqual(len(requests), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
