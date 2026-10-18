"""Full-shape, fixed-weight head-only CUDA Graph benchmark and exactness gate.

Inputs are synthetic, discarded at process exit. This is an implementation
microbenchmark, not native acceptance length or end-to-end throughput evidence.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent / 'native_runtime'))

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from repair.model import DualFeedbackRepair
from head_latency.optimized import VARIANTS, optimized_rollout, prepare_variant
from runtime.precision import build_token_basis
from runtime.utils import dump, sha, now


def load_head(p):
    cfg = p['architecture_by_arm']['BASE_HEAD']
    ck = p['native_checkpoints']['BASE_HEAD']
    assert sha(ck['path']) == ck['sha256']
    assert sha(p['basis_path']) == p['basis_sha256']
    head = DualFeedbackRepair(**cfg).to(device='cuda', dtype=torch.bfloat16).eval()
    state = load_file(ck['path'])
    head.load_state_dict({(k[5:] if k.startswith('head.') else k): v for k, v in state.items()}, strict=True)
    head.requires_grad_(False)
    target = Path('/data/weijiahong/huggingface/models/Qwen3-8B')
    index = json.loads((target / 'model.safetensors.index.json').read_text())['weight_map']
    tensors = []
    for name in ('model.embed_tokens.weight', 'lm_head.weight'):
        with safe_open(target / index[name], framework='pt', device='cpu') as f:
            tensors.append(f.get_tensor(name).to(device='cuda', dtype=torch.bfloat16))
    embedding, lm = tensors
    basis = load_file(p['basis_path'])['repair_basis'].to(device='cuda', dtype=torch.bfloat16)
    head.bind_features(embedding.detach(), build_token_basis(embedding, basis))
    output_basis = build_token_basis(lm, basis)
    return head, lm, output_basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    a = ap.parse_args()
    root = a.root.resolve()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    actual_uuid='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    assert torch.cuda.device_count()==1 and actual_uuid==os.environ['REPAIR_GPU_UUID']
    dump(root/'pilot_environment.json',dict(torch_version=torch.__version__,cuda_version=torch.version.cuda,
        gpu_uuid=actual_uuid,physical_gpu=int(os.environ['REPAIR_PHYSICAL_GPU']),model_dtype='bfloat16',
        compile_emulate_precision_casts=True,tf32=False,scope='inference only; no parameters updated'))
    p = json.loads((root / 'native_protocol.json').read_text())
    rows, timings, errors = [], [], []
    start = time.monotonic()
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
    with torch.no_grad():
        head, lm, output_basis = load_head(p)
        preparation = {}
        for variant in VARIANTS:
            try:
                tick = time.monotonic()
                prepare_variant(head, variant)
                preparation[variant] = dict(status='ready', seconds=time.monotonic()-tick)
            except Exception as exc:
                preparation[variant] = dict(status='failed', error=repr(exc))
                errors.append(dict(variant=variant, error=repr(exc), stage='prepare'))
        dump(root / 'pilot_preparation.json', preparation)
        for n,length in ((1,7),(7,11),(40,7),(1,15),(40,11),(40,15)):
            vocab = head.vocab_size
            idx = torch.arange(n, device='cuda', dtype=torch.int32)
            seeds = torch.arange(n, device='cuda', dtype=torch.int64) + 42
            positions = [torch.full((n,), 129+j, device='cuda', dtype=torch.int64) for j in range(length)]
            columns = [torch.full((n,), j, device='cuda', dtype=torch.int32) for j in range(length)]
            for temperature in (0, 1):
                temp = torch.full((n,), float(temperature), device='cuda')
                for seed in (42, 57):
                    torch.manual_seed(seed)
                    raw = (torch.randn(n, length, vocab, device='cuda') * 2).to(torch.bfloat16)
                    hidden = torch.randn(n, length, head.hidden_size, device='cuda', dtype=torch.bfloat16)
                    anchor = torch.randint(vocab, (n,), device='cuda')
                    pristine = raw.clone()
                    def function(variant, q, capture):
                        def choose(z, j):
                            return gumbel_sample(z, idx, temp, seeds, positions[j], apply_temperature=True,
                                output_processed_logits=q, output_processed_logits_col=columns[j], use_fp64=False)
                        return optimized_rollout(head, raw, hidden, lm, output_basis, anchor, choose,
                            capture=capture, variant=variant)
                    q_ref = torch.empty(n, length, vocab, device='cuda', dtype=torch.float32)
                    reference = function('baseline', q_ref, True)
                    torch.cuda.synchronize()
                    for variant in VARIANTS:
                        if preparation[variant]['status'] != 'ready':
                            continue
                        graph = None
                        try:
                            dump(root / 'pilot_status.json', dict(status='checking', variant=variant, batch_size=n, proposals=length, T=temperature, seed=seed, utc=now()))
                            q = torch.empty_like(q_ref)
                            # Compilation and warmup are outside the measured Graph.
                            for _ in range(3):
                                eager = function(variant, q, True)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                actual = function(variant, q, True)
                            graph.replay()
                            torch.cuda.synchronize()
                            checks = dict(logits_bitwise_equal=torch.equal(actual.logits, reference.logits),
                                sampled_ids_bitwise_equal=torch.equal(actual.tokens, reference.tokens),
                                actual_predecessors_bitwise_equal=torch.equal(actual.predecessors, reference.predecessors),
                                verifier_q_bitwise_equal=torch.equal(q, q_ref), raw_unchanged=torch.equal(raw, pristine),
                                graph_eager_equal=all(torch.equal(getattr(actual, k), getattr(eager, k)) for k in ('logits', 'tokens', 'predecessors')))
                            if seed == 42 and length == 15:
                                saved_zero=raw[:,:,0].clone();saved_hidden=hidden.clone();saved_anchor=anchor.clone()
                                raw[:,:,0].fill_(30);hidden.add_(.125);anchor.add_(1).remainder_(vocab)
                                changed_reference=function('baseline',q_ref,True)
                                graph.replay();torch.cuda.synchronize()
                                checks['graph_input_refresh_equal']=all(torch.equal(getattr(actual,k),getattr(changed_reference,k)) for k in ('logits','tokens','predecessors')) and torch.equal(q,q_ref)
                                raw[:,:,0].copy_(saved_zero);hidden.copy_(saved_hidden);anchor.copy_(saved_anchor)
                                reference=function('baseline',q_ref,True)
                                graph.replay();torch.cuda.synchronize()
                                checks['graph_restored_inputs_equal']=all(torch.equal(getattr(actual,k),getattr(reference,k)) for k in ('logits','tokens','predecessors')) and torch.equal(q,q_ref) and torch.equal(raw,pristine)
                                del saved_zero,saved_hidden,saved_anchor,changed_reference
                            passed = all(checks.values())
                            rows.append(dict(variant=variant, batch_size=n, proposals=length, T=temperature, seed=seed, passed=passed,
                                max_logit_abs=float((actual.logits.float()-reference.logits.float()).abs().max()), **checks))
                            print(now(), 'PARITY', rows[-1], flush=True)
                            del graph, actual, eager
                            graph = None
                            # A failed numerical variant never enters timing selection.
                            if passed and seed == 42 and length == 15:
                                for _ in range(3):
                                    function(variant, q, False)
                                torch.cuda.synchronize()
                                graph = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph):
                                    output = function(variant, q, False)
                                for _ in range(10):
                                    graph.replay()
                                torch.cuda.synchronize()
                                samples = []
                                for _ in range(3):
                                    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                                    begin.record()
                                    for _ in range(100):
                                        graph.replay()
                                    end.record(); end.synchronize()
                                    samples.append(begin.elapsed_time(end)/100)
                                timings.append(dict(variant=variant, batch_size=n, proposals=length, T=temperature,
                                    median_ms=statistics.median(samples), repetitions=100, samples_ms=samples))
                                print(now(), 'TIMING', timings[-1], flush=True)
                                del graph, output
                                graph = None
                        except Exception as exc:
                            errors.append(dict(variant=variant, batch_size=n, proposals=length, T=temperature, seed=seed, error=repr(exc)))
                            rows.append(dict(variant=variant, batch_size=n, proposals=length, T=temperature, seed=seed, passed=False, error=repr(exc)))
                            print(now(), 'REJECTED', errors[-1], flush=True)
                            if variant == 'baseline':
                                raise
                        finally:
                            if graph is not None:
                                del graph
                            gc.collect()
                            dump(root / 'pilot_progress.json', dict(rows=rows, timings=timings, errors=errors, utc=now()))
                    del reference, q_ref, raw, hidden, pristine
                    gc.collect()
        assert len([r for r in rows if r['variant']=='baseline' and r['passed']]) == 24
        reference_time = {(x['batch_size'], x['T']):x['median_ms'] for x in timings if x['variant']=='baseline'}
        candidates = []
        for variant in VARIANTS[1:]:
            checks = [r for r in rows if r['variant']==variant]
            timing = [x for x in timings if x['variant']==variant]
            if len(checks) != 24 or not all(r['passed'] for r in checks) or len(timing) != 4:
                continue
            ratios = [reference_time[x['batch_size'], x['T']]/x['median_ms'] for x in timing]
            if min(ratios) >= 1.03:
                candidates.append(dict(variant=variant, geometric_speedup=math.prod(ratios)**.25, speedup_by_case=ratios))
        candidates = [x for x in candidates if x['variant']=='triton_write']
        selected = max(candidates, key=lambda x:x['geometric_speedup']) if candidates else None
        result = dict(status='complete', selected=selected, parity_rows=rows, timing_rows=timings, errors=errors,
            preparation=preparation, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, elapsed_seconds=time.monotonic()-start,
            scope='Synthetic full-shape fixed-weight head-only CUDA Graph; actual lag-1 stock sampling. Not native acceptance or end-to-end TPS evidence. Real-state smoke gates required before formal.',
            checkpoint_sha256=p['native_checkpoints']['BASE_HEAD']['sha256'], basis_sha256=p['basis_sha256'], utc=now())
        dump(root / 'pilot_summary.json', result)
        dump(root / 'pilot_status.json', dict(status='complete', selected=selected, utc=now()))
        print(now(), 'PILOT_COMPLETE', selected, flush=True)


if __name__ == '__main__':
    main()
