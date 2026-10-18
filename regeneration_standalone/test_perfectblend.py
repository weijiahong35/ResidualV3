"""CPU checks for pool isolation, pinned shards and real-tokenizer HTTP."""
import copy
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
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
POOL = HERE / 'open_perfectblend_regen'
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
sys.path.insert(0, str(HERE / '.client_deps'))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(POOL))
import pool
spec = importlib.util.spec_from_file_location('pool_launcher', POOL / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class PoolChecks(unittest.TestCase):
    def test_dedup_benchmark_exclusion_group_split_and_source_sampling(self):
        rows = [{'source': 'math' if i % 3 else 'chat', 'conversations': [
                    {'from': 'human', 'value': f'Question {i}'},
                    {'from': 'gpt', 'value': 'ORIGINAL_ANSWER'}]} for i in range(200)]
        rows.append(copy.deepcopy(rows[0]))
        extra = copy.deepcopy(rows[1])
        extra['conversations'].append({'from': 'human', 'value': 'Different follow-up'})
        rows.append(extra)
        data = rows
        records, groups, counts, skipped, _ = pool.eligible_rows(data, {pool.prompt_key('  Question\n199 ')})
        self.assertEqual(skipped['benchmark_prompt_match'], 1)
        self.assertEqual(skipped['duplicate_conversation_prompt'], 1)
        self.assertEqual(len(records), 200)
        train, validation = pool.partition(records, groups, 42, 0.2)
        self.assertFalse({row[2] for row in train} & {row[2] for row in validation})
        self.assertEqual(any(row[0] == 1 for row in train), any(row[0] == 201 for row in train))
        chunks, quota = pool.select_shards(train, 100, 25, 42)
        self.assertEqual([len(chunk) for chunk in chunks], [25] * 4)
        selected = [index for chunk in chunks for index in chunk]
        self.assertEqual(len(set(selected)), 100)
        from collections import Counter
        self.assertEqual(Counter(rows[index]['source'] for index in selected), quota)
        self.assertEqual(chunks, pool.select_shards(train, 100, 25, 42)[0])
        messages, users, _, _ = pool.prompt_only(extra)
        self.assertEqual([message['role'] for message in messages], ['user', 'user'])
        self.assertNotIn('ORIGINAL_ANSWER', json.dumps(messages))

    def test_shard_selection_and_explicit_gpu(self):
        self.assertEqual(launcher.shard_numbers('0-2,4,2', range(5)), [0, 1, 2, 4])
        with self.assertRaises(ValueError):
            launcher.shard_numbers('4-2', range(5))
        with self.assertRaises(ValueError):
            launcher.shard_numbers('5', range(5))
        result = subprocess.run([sys.executable, str(POOL / 'run.py'), '--run'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires explicit --gpu', result.stderr)

    def test_prompt_files_http_metadata_thinking_multiturn_resume_and_plan(self):
        self._check_http_regeneration('on')

    def test_nonthinking_4096_multiturn_budget_and_resume(self):
        self._check_http_regeneration('off')

    def _check_http_regeneration(self, thinking):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(launcher.backend.MODELS / 'Qwen3-8B'))
        requests = []
        def render(payload):
            return tokenizer.apply_chat_template(payload['messages'], tokenize=True,
                add_generation_prompt=True, return_dict=False, **payload.get('chat_template_kwargs', {}))
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                prompt_ids = render(payload)
                if self.path == '/tokenize':
                    answer = {'count': len(prompt_ids), 'tokens': prompt_ids}
                else:
                    requests.append(copy.deepcopy(payload))
                    text = ('<think>Regenerated reasoning.</think>\nRegenerated answer.'
                            if thinking == 'on' else 'Regenerated answer without a reasoning trace.')
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    answer = {'prompt_token_ids': prompt_ids, 'choices': [
                        {'message': {'role': 'assistant', 'content': text}, 'token_ids': ids, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': len(prompt_ids), 'completion_tokens': len(ids)}}
                body = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                raw = directory / 'prompts/train/shard-000.jsonl'
                raw.parent.mkdir(parents=True)
                revision = 'a' * 40
                row = {'id': 'opb:aaaaaaaaaaaa:0000001', 'source': 'original/mixed-source',
                       'conversations': [{'role': 'user', 'content': 'First user prompt'},
                                         {'role': 'user', 'content': 'Second user prompt'}],
                       'pool': {'revision': revision, 'original_row_index': 1, 'split': 'train', 'shard': 0}}
                raw.write_text(json.dumps(row) + '\n')
                pool.write_json(directory / 'manifest.json', {'parameters': {'revision': revision, 'seed': 42},
                    'shards': [{'split': 'train', 'shard': 0, 'rows': 1, 'path': 'prompts/train/shard-000.jsonl',
                                'sha256': pool.sha256_file(raw)}]})
                with patch.object(sys, 'argv', ['run.py', '--pool-dir', str(directory), '--shards', '0',
                                               '--thinking', thinking]):
                    args = launcher.parse_args()
                    runtime, plan = launcher.make_plan(args)
                self.assertEqual(plan['mode'], 'dry_run_no_gpu_process')
                self.assertEqual(plan['target_conversations'], 1)
                self.assertEqual(plan['contract']['sampling_params']['temperature'], 0.6 if thinking == 'on' else 0.7)
                self.assertEqual(plan['contract']['sampling_params']['seed'], 42)
                command = plan['jobs'][0]['client_argv']
                command[command.index('--endpoint') + 1] = f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
                command += ['--limit', '1', '--summary-file', str(directory / 'summary.json')]
                result = subprocess.run(command, env=dict(os.environ, PYTHONNOUSERSITE='1'),
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                output = Path(plan['jobs'][0]['output'])
                samples = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual(len(samples), 2)
                self.assertEqual(len(requests), 2)
                for sample in samples:
                    self.assertEqual(sample['metadata']['source']['original_source'], row['source'])
                    self.assertEqual(sample['metadata']['source']['pool'], row['pool'])
                    self.assertEqual(sample['metadata']['source']['revision'], revision)
                    if thinking == 'on':
                        self.assertIn('Regenerated reasoning.</think>', sample['text'])
                    else:
                        self.assertNotIn('Regenerated reasoning', sample['text'])
                        self.assertLessEqual(len(sample['input_ids']), 4096)
                    self.assertEqual(len(sample['input_ids']), len(sample['loss_mask']))
                    self.assertTrue(any(sample['loss_mask']))
                self.assertEqual([message['role'] for message in requests[1]['messages']], ['user', 'assistant', 'user'])
                for payload in requests:
                    self.assertEqual(payload['chat_template_kwargs']['enable_thinking'], thinking == 'on')
                    self.assertEqual(payload['max_completion_tokens'],
                                     (8192 if thinking == 'on' else 4096) - len(render(payload)))
                self.assertEqual(launcher.backend.count_existing(output), 1)
                self.assertEqual(launcher.completed_count(plan['jobs'][0]), 1)
                before = output.read_bytes()
                result = subprocess.run(command, env=dict(os.environ, PYTHONNOUSERSITE='1'),
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(len(requests), 2)
                foreign = copy.deepcopy(samples[0])
                foreign['primary_id'] = 'foreign-conversation'
                output.write_text(json.dumps(foreign) + '\n')
                with self.assertRaisesRegex(ValueError, 'does not belong'):
                    launcher.completed_count(plan['jobs'][0])
                raw.write_text(raw.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'shard changed'):
                    launcher.make_plan(args)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
