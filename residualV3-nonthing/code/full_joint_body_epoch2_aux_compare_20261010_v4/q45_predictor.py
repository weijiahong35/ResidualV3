"""Detached post-repair q statistics; auxiliary parameters have their own Adam.

The 45 features reproduce q_probe_capture.features: four confidence statistics,
nine cumulative top probabilities, and 32 sorted log probabilities. No token IDs.
The two outputs predict T0 argmax agreement and T1 analytical overlap. Training
labels are teacher-forced proxies, not measured native first-rejection events.
"""
import math
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _normalizer_entropy(X, S, R: tl.constexpr, V: tl.constexpr,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    maximum = float('-inf')
    denominator = 0.
    weighted_logits = 0.
    for start in range(0, V, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        valid = offsets < V
        x = tl.load(X + row * V + offsets, valid, other=float('-inf')).to(tl.float32)
        next_max = tl.maximum(maximum, tl.max(x, 0))
        factor = tl.exp(maximum - next_max)
        mass = tl.where(valid, tl.exp(x - next_max), 0.)
        denominator = denominator * factor + tl.sum(mass, 0)
        weighted_logits = weighted_logits * factor + tl.sum(mass * tl.where(valid, x, 0.), 0)
        maximum = next_max
    lse = maximum + tl.log(denominator)
    entropy = (lse - weighted_logits / denominator) / tl.log(V * 1.0)
    tl.store(S + row, lse)
    tl.store(S + R + row, entropy)


@torch.no_grad()
def features(logits):
    """Unit-temperature full-vocabulary q, without a dense FP32 probability copy."""
    logits = logits.detach().contiguous()
    shape = logits.shape[:-1]
    rows, vocab = logits.numel() // logits.shape[-1], logits.shape[-1]
    assert vocab >= 256 and logits.dtype == torch.bfloat16
    stats = torch.empty((2, rows), device=logits.device, dtype=torch.float32)
    _normalizer_entropy[(rows,)](logits, stats, rows, vocab, 8192, num_warps=8)
    top_values, top_ids = logits.reshape(rows, vocab).topk(256, -1)
    log_top = top_values.float() - stats[0, :, None]
    probabilities = log_top.exp()
    cumulative = probabilities.cumsum(-1)
    indices = torch.tensor([0, 1, 3, 7, 15, 31, 63, 127, 255], device=logits.device)
    four = torch.stack((probabilities[:, 0], probabilities[:, 1],
                        probabilities[:, 0] - probabilities[:, 1], stats[1]), -1)
    values = torch.cat((four, cumulative[:, indices],
                        log_top[:, :32].clamp_min(math.log(1e-30))), -1)
    return values.reshape(*shape, 45), top_ids[:, 0].reshape(shape)


@torch.no_grad()
def reference_features(logits):
    q = logits.float().softmax(-1)
    top, _ = q.topk(256, -1)
    entropy = -(q * q.clamp_min(1e-30).log()).sum(-1) / math.log(q.shape[-1])
    four = torch.stack((top[..., 0], top[..., 1], top[..., 0] - top[..., 1], entropy), -1)
    cumulative = torch.stack([top[..., :k].sum(-1) for k in (1, 2, 4, 8, 16, 32, 64, 128, 256)], -1)
    return torch.cat((four, cumulative, top[..., :32].clamp_min(1e-30).log()), -1)


class Predictor(nn.Module):
    def __init__(self, device):
        super().__init__()
        # No random initialization: adding the branch must not consume main RNG.
        self.weight = nn.Parameter(torch.zeros(2, 45, device=device))
        self.bias = nn.Parameter(torch.zeros(2, device=device))
        self.register_buffer('mean', torch.zeros(45, device=device))
        self.register_buffer('scale', torch.ones(45, device=device))
        self.register_buffer('initialized', torch.tensor(False, device=device))

    def forward(self, x):
        x = x.detach().float()
        return torch.nn.functional.linear((x - self.mean) / self.scale, self.weight, self.bias)


class Auxiliary:
    def __init__(self, device='cuda', lr=0.003):
        self.model = Predictor(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.001)
        self.steps = 0

    def train(self, x, labels):
        x = x.detach().reshape(-1, 45)
        labels = labels.detach().float().reshape(-1, 2)
        if self.steps == 0:
            with torch.no_grad():
                self.model.mean.copy_(x.mean(0))
                self.model.scale.copy_(x.std(0, unbiased=False).clamp_min(0.01))
                self.model.initialized.fill_(True)
        self.optimizer.zero_grad(set_to_none=True)
        prediction = self.model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, labels)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        self.optimizer.step()
        self.steps += 1
        return dict(loss=float(loss.detach()), gradient_norm=float(norm), steps=self.steps)

    def state_dict(self):
        return dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), steps=self.steps,
                    feature_count=45, output_labels=['T0_argmax_agreement', 'T1_overlap'], input='post_repair_q')

    def load_state_dict(self, state):
        assert state['feature_count'] == 45 and state['input'] == 'post_repair_q'
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.steps = state['steps']
