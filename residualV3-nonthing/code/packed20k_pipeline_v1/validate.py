"""Bounded real-pack parity/gradient/cost checks; zero optimizer updates."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parent
ORIGINAL = CODE.parent / 'packed20k_split_v3'
sys.path.insert(0, str(ORIGINAL / 'runtime_bundle'))
sys.path.insert(0, str(ORIGINAL / 'runtime_bundle/src'))
sys.path.insert(0, str(ORIGINAL / 'runtime_bundle/hs_connectors/src'))
sys.path.insert(0, str(CODE.parent / 'packed20k_recovery_v1'))
import torch
from safetensors.torch import load_file
from runtime.utils import dump, sha, now
from runtime.live_trainer import load_models
from runtime.precision import build_token_basis
from repair.model import DualFeedbackRepair, rollout, output_loss
from transport import install
from pipeline import PackPipeline
from compute import sharded_states


def verify_pipeline():
    train = [{'input_ids': [i]} for i in range(12)]
    packs = [list(range(i, i + 3)) for i in range(0, 12, 3)]
    class Provider:
        def begin_epoch(self, rows):
            self.order = iter(row['input_ids'][0] for row in rows)
        def __call__(self, row):
            assert row['input_ids'][0] == next(self.order)
            return row['input_ids'][0]
    pipe = PackPipeline(Provider(), train, packs)
    for expected in packs:
        indices, _, chunks = pipe.get()
        assert indices == chunks == expected
        assert pipe.peak_resident <= 2
        pipe.release()
    pipe.close()
    assert not pipe.thread.is_alive()
    class FailedProvider(Provider):
        def __call__(self, row):
            if row['input_ids'][0] == 3:
                raise ValueError('synthetic transport failure')
            return super().__call__(row)
    pipe = PackPipeline(FailedProvider(), train, packs)
    assert pipe.get()[0] == packs[0]
    pipe.release()
    try:
        pipe.get()
    except ValueError:
        pass
    else:
        raise AssertionError('producer failure swallowed')
    pipe.close()
    pipe = PackPipeline(Provider(), train, packs)
    pipe.close()
    assert not pipe.thread.is_alive()
    return dict(order_exact=True, resident_packs_bounded_by_two=True,
                producer_failure_propagated=True, cancellation_joined=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cpu = verify_pipeline()
    dump(out / 'pipeline_cpu_checks.json', cpu)
    if args.cpu_only:
        print(json.dumps(cpu), flush=True)
        return
    p = json.loads((args.run / 'training_protocol.json').read_text())
    install(out / 'hidden_connection_retries.jsonl')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target, draft = load_models(torch.device('cuda:0'), f"http://127.0.0.1:{p['port']}/v1",
                               p['transfer_root'], 16)
    provider = target.hidden_provider
    spec = importlib.util.spec_from_file_location('frozen_original_train', ORIGINAL / 'train.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    train = [json.loads(line) for line in Path(p['data_source']).read_text().splitlines()]
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    packs = original.packs_for_epoch(train, p['experiment_seed'], 2, 65536)
    indices = packs[state['step'] % len(packs)]
    rows = [train[index] for index in indices]
    basis = load_file(p['basis_path'])['repair_basis'].to('cuda', dtype=torch.bfloat16)
    output_basis = build_token_basis(target.lm_head.weight, basis)
    head = DualFeedbackRepair(**p['architecture']).to('cuda', dtype=torch.bfloat16)
    head.bind_features(target.model.embed_tokens.weight, build_token_basis(target.model.embed_tokens.weight, basis))
    head.load_state_dict(state['head'])
    reference_head = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    checkpoint_step = state['step']
    del state
    def timed(function):
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        return result, dict(seconds=time.perf_counter() - begin,
            extra_peak_allocated_bytes=torch.cuda.max_memory_allocated() - baseline_memory)
    try:
        provider.begin_epoch(rows)
        chunks = [provider(row) for row in rows]
        torch.manual_seed(p['experiment_seed'] + checkpoint_step + 1 + 200000)
        reference, _ = timed(lambda: original.packed_states(target, draft, rows, chunks, 65536, 1536))
        checks = {}
        for shard in (512, 256):
            candidate, cold = timed(lambda: sharded_states(target, draft, rows, chunks, 65536, 1536,
                                   reference[4].tolist(), shard_anchors=shard))
            assert all(torch.equal(a, b) for a, b in zip(reference[1:], candidate[1:]))
            relative = float((reference[0].float() - candidate[0].float()).norm() /
                             reference[0].float().norm().clamp_min(1e-12))
            assert relative < .02, (shard, relative)
            del candidate
            _, warm = timed(lambda: sharded_states(target, draft, rows, chunks, 65536, 1536,
                            reference[4].tolist(), shard_anchors=shard))
            checks[str(shard)] = dict(exact_sample_and_teacher_alignment=True,
                                      draft_hidden_relative_l2=relative, cold=cold, warm=warm)
            dump(out / 'draft_shard_checks.json', checks)
            print('SHARD_CHECK', shard, json.dumps(checks[str(shard)]), flush=True)
        _, reference_timing = timed(lambda: original.packed_states(target, draft, rows, chunks, 65536, 1536,
                                                                   reference[4].tolist()))
        sharded, _ = timed(lambda: sharded_states(target, draft, rows, chunks, 65536, 1536, reference[4].tolist(), shard_anchors=512))
        h, th, answers, anchor = [tensor[:64] for tensor in reference[:4]]
        with torch.no_grad():
            raw, gold = h @ target.lm_head.weight.T, th @ target.lm_head.weight.T
        head.zero_grad(set_to_none=True)
        old_result = rollout(head, raw, h, target.lm_head.weight, output_basis, anchor,
                             lambda z, j: answers[:, j])
        old_logits = old_result.logits.detach()
        old_loss = output_loss(old_result.logits, gold)
        old_loss.backward()
        gradients = torch.cat([parameter.grad.detach().float().flatten()
                              for parameter in head.parameters() if parameter.grad is not None])
        old_loss_value = float(old_loss.detach())
        del old_result, old_loss
        head.zero_grad(set_to_none=True)
        new_h, new_th, new_answers, new_anchor = [tensor[:64] for tensor in sharded[:4]]
        with torch.no_grad():
            new_raw, new_gold = new_h @ target.lm_head.weight.T, new_th @ target.lm_head.weight.T
        new_result = rollout(head, new_raw, new_h, target.lm_head.weight, output_basis, new_anchor,
                             lambda z, j: new_answers[:, j])
        new_logits = new_result.logits
        new_loss = output_loss(new_logits, new_gold)
        new_loss.backward()
        new_gradients = torch.cat([parameter.grad.detach().float().flatten()
                                  for parameter in head.parameters() if parameter.grad is not None])
        relative_grad = float((gradients - new_gradients).norm() / gradients.norm().clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(gradients[None], new_gradients[None]))
        agreement = float((old_logits.argmax(-1) == new_logits.argmax(-1)).float().mean())
        loss_diff = abs(float(new_loss.detach()) - old_loss_value)
        dump(out / 'unchanged_head_gradient_check.json', dict(relative_gradient_l2=relative_grad, gradient_cosine=cosine, argmax_agreement=agreement, loss_absolute_difference=loss_diff))
        assert relative_grad < .02 and cosine > .999 and agreement == 1.0 and loss_diff < .001
        assert all(torch.equal(value.detach().cpu(), reference_head[name]) for name, value in head.state_dict().items())
        report = dict(status='passed', checkpoint_step=checkpoint_step, optimizer_updates=0,
            head_parameters_unchanged=True, pipeline=cpu, draft_shards=checks,
            original_draft_warm=reference_timing,
            unchanged_head_on_sharded_states=dict(relative_gradient_l2=relative_grad, gradient_cosine=cosine,
                argmax_agreement=agreement, loss_absolute_difference=loss_diff, anchors=64),
            scope='one actual packed input, 1536 sampled anchors; 64-anchor head forward/gradient check; not full native bitwise parity',
            gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'], checkpoint_sha256=sha(args.checkpoint), utc=now())
        dump(out / 'validation.json', report)
        print(json.dumps(report), flush=True)
    finally:
        provider.close()
        dump(out / 'hidden_cleanup.json', dict(requests=provider.requests, deleted=provider.deleted,
             equal=provider.requests == provider.deleted, no_persistent_hidden_cache=True, utc=now()))


if __name__ == '__main__':
    main()
