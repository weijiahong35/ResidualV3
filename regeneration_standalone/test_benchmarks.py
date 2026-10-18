"""CPU checks using downloaded benchmarks, a real tokenizer and mock HTTP."""
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
sys.path.insert(0, str(HERE / '.client_deps'))
sys.path.insert(0, str(HERE))
from regenlib.configs import DATASET_CONFIGS
import run


class BenchmarkChecks(unittest.TestCase):
    def test_presets_and_no_reference_leakage(self):
        self.assertEqual((HERE / 'regenlib/configs.py').read_bytes(),
                         (HERE.parent / 'src/speculators/data_generation/configs.py').read_bytes())
        self.assertEqual(DATASET_CONFIGS['gsm8k'].split, 'train')
        self.assertEqual(DATASET_CONFIGS['gsm8k_test'].split, 'test')
        cfg = DATASET_CONFIGS['humaneval']
        row = {'prompt': 'def add(a, b):\n    """Add numbers."""',
               'canonical_solution': 'SECRET_SOLUTION', 'test': 'SECRET_TESTS'}
        prompt = cfg.normalize_fn(row)['conversations'][0]['content']
        self.assertIn(row['prompt'], prompt)
        self.assertNotIn('SECRET_', prompt)
        turns = DATASET_CONFIGS['mt_bench'].normalize_fn({'prompt': ['first', 'second']})['conversations']
        self.assertEqual([t['content'] for t in turns], ['first', 'second'])
        with self.assertRaises(ValueError):
            DATASET_CONFIGS['mt_bench'].normalize_fn({'prompt': 'not a turn list'})

    def test_launcher_split_mode_and_counts(self):
        for dataset, split, count in [('gsm8k', 'train', 7473), ('humaneval', 'test', 164), ('mt_bench', 'train', 80)]:
            with self.subTest(dataset=dataset), patch.object(sys, 'argv', ['run.py', '--dataset', dataset]):
                args = run.parse_args()
                plan = run.make_plan(args)
                self.assertFalse(args.run)
                self.assertEqual(args.split, split)
                self.assertEqual(args.max_samples, count)
                self.assertTrue(plan['sampling_params']['chat_template_kwargs']['enable_thinking'])
                self.assertIn(f'_{split}_thinking_', plan['output'])
                self.assertEqual(plan['dataset_role'], 'benchmark_train_split' if dataset == 'gsm8k' else 'evaluation')
        with patch.object(sys, 'argv', ['run.py', '--dataset', 'gsm8k', '--split', 'test']):
            args = run.parse_args()
            plan = run.make_plan(args)
            self.assertEqual(args.max_samples, 1319)
            self.assertEqual(plan['dataset_preset'], 'gsm8k_test')
            self.assertEqual(plan['dataset_role'], 'evaluation')
        with patch.object(sys, 'argv', ['run.py', '--dataset', 'gsm8k', '--split', 'train', '--thinking', 'off']):
            args = run.parse_args()
            plan = run.make_plan(args)
            self.assertEqual(args.max_samples, 7473)
            self.assertIn('_train_nonthinking_', plan['output'])
            self.assertFalse(plan['sampling_params']['chat_template_kwargs']['enable_thinking'])
        with patch.object(sys, 'argv', ['run.py', '--dataset', 'opencodeinstruct']):
            args = run.parse_args()
            self.assertEqual(args.max_samples, 10000)
            self.assertEqual(args.sampling_params, {})
            self.assertTrue(run.make_plan(args)['output'].endswith('/opencode_qwen3_8b_onpolicy.jsonl'))

    def test_cached_benchmarks_http_thinking_multiturn_and_resume(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(run.MODELS / 'Qwen3-8B'))
        manifest = json.loads((HERE / 'download_manifest.json').read_text())
        entries = [e for e in manifest['datasets'] if e['dataset'] != 'gsm8k' or e['split'] == 'test']
        state = {'requests': []}

        def render(payload):
            return tokenizer.apply_chat_template(payload['messages'], tokenize=True,
                add_generation_prompt=True, return_dict=False,
                **payload.get('chat_template_kwargs', {}))

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                prompt_ids = render(payload)
                if self.path == '/tokenize':
                    answer = {'count': len(prompt_ids), 'tokens': prompt_ids}
                else:
                    state['requests'].append(copy.deepcopy(payload))
                    content = '<think>CPU mock reasoning.</think>\nMock answer.'
                    completion_ids = tokenizer.encode(content, add_special_tokens=False)
                    answer = {'prompt_token_ids': prompt_ids, 'choices': [
                        {'message': {'role': 'assistant', 'content': content},
                         'token_ids': completion_ids, 'finish_reason': 'stop'}],
                        'usage': {'prompt_tokens': len(prompt_ids), 'completion_tokens': len(completion_ids)}}
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
                for entry in entries:
                    with self.subTest(dataset=entry['dataset']):
                        raw = load_dataset(entry['hf_path'], name=entry['subset'], split=entry['split'], revision=entry['revision'])
                        self.assertEqual(len(raw), entry['rows'])
                        state['requests'].clear()
                        preset = 'gsm8k_test' if entry['dataset'] == 'gsm8k' else entry['dataset']
                        expected_turns = DATASET_CONFIGS[preset].normalize_fn(raw[0])['conversations']
                        expected_turns = [t for t in expected_turns if t['role'] == 'user']
                        output = Path(tmp) / (preset + '.jsonl')
                        command = [sys.executable, str(HERE / 'regenerate.py'), '--dataset', preset,
                            '--dataset-revision', entry['revision'], '--model', str(run.MODELS / 'Qwen3-8B'),
                            '--endpoint', f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                            '--outfile', str(output), '--resume', '--limit', '1', '--concurrency', '1',
                            '--max-retries', '0',
                            '--max-model-len', '8192', '--max-tokens', '1024',
                            '--sampling-params', json.dumps({'temperature': 0.6, 'chat_template_kwargs': {'enable_thinking': True}})]
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1')
                        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        samples = [json.loads(line) for line in output.read_text().splitlines()]
                        self.assertEqual(len(samples), len(expected_turns))
                        for i, (sample, payload) in enumerate(zip(samples, state['requests'])):
                            self.assertEqual(payload['messages'][-1]['content'], expected_turns[i]['content'])
                            prompt_ids = render(payload)
                            self.assertEqual(sample['input_ids'][:len(prompt_ids)], prompt_ids)
                            self.assertEqual(sample['loss_mask'][:len(prompt_ids)], [0] * len(prompt_ids))
                            self.assertEqual(sample['loss_mask'][len(prompt_ids):], [1] * (len(sample['input_ids']) - len(prompt_ids)))
                            self.assertEqual(sample['text'], tokenizer.decode(sample['input_ids'], skip_special_tokens=False))
                            self.assertIn('<think>', sample['text'])
                            self.assertIn('CPU mock reasoning.</think>', sample['text'])
                            self.assertEqual(sample['metadata']['source']['revision'], entry['revision'])
                        if entry['dataset'] == 'humaneval':
                            self.assertEqual(samples[0]['primary_id'], raw[0]['task_id'])
                            self.assertNotIn(raw[0]['canonical_solution'], state['requests'][0]['messages'][0]['content'])
                        if entry['dataset'] == 'mt_bench':
                            self.assertEqual([m['role'] for m in state['requests'][1]['messages']], ['user', 'assistant', 'user'])
                            self.assertEqual(state['requests'][1]['messages'][1]['content'], '<think>CPU mock reasoning.</think>\nMock answer.')
                            original = output.read_bytes()
                            # Resume skips the whole completed conversation and
                            # produces both turns of the next one, with no rewrite.
                            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertTrue(output.read_bytes().startswith(original))
                            resumed = [json.loads(line) for line in output.read_text().splitlines()]
                            self.assertEqual(len(resumed), 4)
                            self.assertEqual(len({s['primary_id'] for s in resumed}), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main(verbosity=2)
