"""Kill a real client between turns and verify durable resume without replay."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from regenlib.checkpoint import repair_output_tail


class DurableResumeChecks(unittest.TestCase):
    def test_live_concurrency_increase_and_non_destructive_decrease(self):
        import asyncio
        import regenerate
        async def check():
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'concurrency.json'
                path.write_text(json.dumps({'concurrency': 2}))
                gate = regenerate.LiveConcurrency(path, 128)
                await gate.acquire()
                await gate.acquire()
                pending = asyncio.create_task(gate.acquire())
                await asyncio.sleep(.25)
                self.assertFalse(pending.done())
                path.write_text(json.dumps({'concurrency': 3}))
                await asyncio.wait_for(pending, 1)
                self.assertEqual(gate.active, 3)
                path.write_text(json.dumps({'concurrency': 1}))
                pending = asyncio.create_task(gate.acquire())
                await asyncio.sleep(.25)
                self.assertFalse(pending.done())
                self.assertEqual(gate.active, 3)  # active generations survive
                gate.active = 0
                await asyncio.wait_for(pending, 1)
                self.assertEqual(gate.active, 1)
                path.write_text(json.dumps({'concurrency': 129}))
                with self.assertRaises(ValueError):
                    gate.limit()
        asyncio.run(check())

    def test_sigkill_between_turns_and_torn_output_append(self):
        requests = []
        release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    result = {'count': 3, 'tokens': [1, 2, 3]}
                else:
                    requests.append(payload)
                    if len(requests) == 2:
                        release.wait(30)
                    result = {'prompt_token_ids': [1, 2, 3], 'choices': [{
                        'message': {'role': 'assistant', 'content': 'saved answer'},
                        'token_ids': [6, 7], 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        child = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                prompt = directory / 'prompts.jsonl'
                prompt.write_text(json.dumps({'id': 'two-turns', 'conversations': [
                    {'role': 'user', 'content': 'first'}, {'role': 'user', 'content': 'second'}]}) + '\n')
                output = directory / 'generated.jsonl'
                command = [sys.executable, str(HERE / 'regenerate.py'), '--dataset', 'open-perfectblend',
                    '--dataset-files', str(prompt), '--model', '/data/weijiahong/huggingface/models/Qwen3-8B',
                    '--endpoint', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                    '--outfile', str(output), '--resume', '--limit', '1', '--concurrency', '1',
                    '--max-model-len', '8192', '--max-tokens', '1024', '--max-retries', '0']
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', USE_TORCH='0', USE_TF='0', USE_FLAX='0',
                           HF_HUB_OFFLINE='1', PYTHONNOUSERSITE='1')
                with (directory / 'first.log').open('wb') as log:
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 45
                    while len(requests) < 2 and time.monotonic() < deadline and child.poll() is None:
                        time.sleep(.05)
                    self.assertEqual(len(requests), 2, (directory / 'first.log').read_text())
                    checkpoint_files = list(Path(str(output) + '.checkpoints').glob('*.json'))
                    self.assertEqual(len(checkpoint_files), 1)
                    saved = json.loads(checkpoint_files[0].read_text())['progress']
                    self.assertEqual(saved['next_turn'], 1)
                    self.assertEqual(len(saved['samples']), 1)
                    child.kill()
                    child.wait(timeout=10)
                # Simulate a kill during the final multi-row append as well.
                first = saved['samples'][0]
                first['metadata']['conversation_complete'] = False
                output.write_bytes((json.dumps(first) + '\n{"primary_id":').encode())
                release.set()
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(rows), 2)
                self.assertEqual([r['id'] for r in rows], ['two-turns_gen0', 'two-turns_gen1'])
                self.assertEqual(rows[0]['input_ids'], saved['samples'][0]['input_ids'])
                self.assertEqual(sum(p['messages'][-1]['content'] == 'first' for p in requests), 1)
                self.assertEqual([p['messages'][-1]['content'] for p in requests], ['first', 'second', 'second'])
                self.assertEqual(requests[-1]['messages'][1]['content'], 'saved answer')
                self.assertTrue(rows[-1]['metadata']['conversation_complete'])
                self.assertFalse(list(Path(str(output) + '.checkpoints').glob('*.json')))
                self.assertEqual(len(list(directory.glob('*.interrupted-tail-*.bin'))), 1)
                self.assertIsNone(repair_output_tail(output))
                before = output.read_bytes()
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(len(requests), 3)
        finally:
            release.set()
            if child and child.poll() is None:
                child.kill()
                child.wait()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
