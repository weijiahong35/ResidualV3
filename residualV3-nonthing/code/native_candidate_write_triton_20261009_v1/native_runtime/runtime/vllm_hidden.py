"""ResidualV2-style online vLLM extraction with bounded RAM-only transfer."""
import json
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from threading import Lock
from pathlib import Path
import torch
from safetensors import safe_open


def require_tmpfs(path):
    path = Path(path).resolve()
    # Accept only the known RAM mount, never /tmp or a dataset directory.
    if Path('/dev/shm') not in path.parents:
        raise RuntimeError('hidden transfer must be under /dev/shm')
    mounts = [line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    if not any(row[1] == '/dev/shm' and row[2] == 'tmpfs' for row in mounts):
        raise RuntimeError('/dev/shm is not tmpfs')
    return path


class OnlineHiddenProvider:
    """Bounded concurrent prefetch in training order; no cross-update cache."""
    def __init__(self, endpoint, root, model, layers=6, width=4096, prefetch=4):
        import openai
        from hs_connectors import FileTransfer
        self.root = require_tmpfs(root)
        if not self.root.is_dir(): raise RuntimeError('server transfer directory missing')
        self.client = openai.OpenAI(base_url=endpoint, api_key='EMPTY', max_retries=0)
        actual = self.client.models.list().data[0].id
        if actual != model: raise RuntimeError(f'target server model mismatch: {actual}')
        self.model, self.layers, self.width = model, layers, width
        self.transfer = FileTransfer(self.root)
        if not 1 <= prefetch <= 16: raise ValueError('bounded prefetch must be 1..16')
        self.prefetch = prefetch
        self.pool = ThreadPoolExecutor(max_workers=prefetch)
        self.queue = deque()
        self.source = iter(())
        self.counter_lock = Lock()
        self.requests = 0
        self.deleted = 0
        self.inflight = 0
        self.peak_inflight = 0

    def fetch(self, example):
        with self.counter_lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            return self._fetch(example)
        finally:
            with self.counter_lock: self.inflight -= 1

    def _fetch(self, example):
        from speculators.data_generation.vllm_client import generate_hidden_states
        handle = generate_hidden_states(self.client, self.model,
                    {'input_ids': example['input_ids']}, timeout=180, max_retries=0)
        path = Path(handle).resolve()
        if path.parent != self.root: raise RuntimeError(f'wrong hidden handle root: {path}')
        try:
            payload = self.transfer.get_generated(str(path))
            if payload is None: raise RuntimeError('missing generated hidden states')
            ids, h = payload['token_ids'], payload['hidden_states']
            if ids.tolist() != example['input_ids']: raise RuntimeError('target token alignment mismatch')
            if h.shape != (len(example['input_ids']), self.layers, self.width):
                raise RuntimeError(f'wrong hidden shape: {tuple(h.shape)}')
            if h.dtype != torch.bfloat16 or not torch.isfinite(h).all():
                raise RuntimeError('hidden dtype/nonfinite error')
            # Own RAM copy before unlinking the mmap-backed transport file.
            result = h.clone()
            with self.counter_lock: self.requests += 1
            return result
        finally:
            path.unlink(missing_ok=True)
            Path(str(path)+'.lock').unlink(missing_ok=True)
            with self.counter_lock: self.deleted += 1

    def _fill(self):
        while len(self.queue) < self.prefetch:
            example = next(self.source, None)
            if example is None: break
            self.queue.append((example['input_ids'], self.pool.submit(self.fetch, example)))

    def begin_epoch(self, examples):
        if self.queue: raise RuntimeError('unconsumed hidden from previous epoch')
        self.source = iter(examples)
        self._fill()

    def __call__(self, example):
        if not self.queue: raise RuntimeError('call begin_epoch with the exact training order')
        ids, future = self.queue.popleft()
        if ids != example['input_ids']: raise RuntimeError('prefetch order mismatch')
        h = future.result()
        self._fill()
        return h

    def close(self):
        # Finish/unlink in-flight transfers, cancel queued work, then release RAM.
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.queue.clear()
        self.source = iter(())
        self.client.close()


def load_target_components(model_dir, device, endpoint, root, prefetch=4):
    """Load only original frozen embedding/head/norm, not a second target body."""
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    model_dir = Path(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    index = json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    def tensor(name):
        with safe_open(model_dir/index[name], framework='pt', device='cpu') as f:
            return f.get_tensor(name).to(device=device, dtype=torch.bfloat16)
    target = torch.nn.Module()
    target.model = torch.nn.Module()
    target.model.embed_tokens = torch.nn.Embedding.from_pretrained(tensor('model.embed_tokens.weight'), freeze=True)
    with torch.device('meta'):
        target.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    target.lm_head.weight = torch.nn.Parameter(tensor('lm_head.weight'), requires_grad=False)
    target.model.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad(): target.model.norm.weight.copy_(tensor('model.norm.weight'))
    target.requires_grad_(False)
    target.hidden_provider = OnlineHiddenProvider(endpoint, root, str(model_dir), prefetch=prefetch)
    return target.eval()
