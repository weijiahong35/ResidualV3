"""Same frozen models/head as the per-sequence 20k arm; whole-pack sampling."""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'runtime_bundle'))
import torch
from safetensors.torch import load_file, save_file
from repair.model import DualFeedbackRepair, rollout, output_loss
from repair.run import digest
from repair.recovery_state import save_checkpoint
from runtime.live_trainer import load_models, online_states, schedule_scale
from runtime.live_run import order_for_epoch
from runtime.master_optimizer import MasterAdamW
from runtime.precision import build_token_basis
from runtime.utils import dump, sha, now
from speculators.models.dflash.utils import select_anchors, get_base_indices_for_anchored_blocks


def packs_for_epoch(train, seed, epoch, budget):
    packs, rows, used = [], [], 0
    for index in order_for_epoch(len(train), seed, epoch):
        length = len(train[index]['input_ids'])
        assert 0 < length <= 4096
        if rows and used + length > budget:
            packs.append(rows)
            rows, used = [], 0
        rows.append(index)
        used += length
    if rows:
        packs.append(rows)
    return packs


def eligible_positions(row):
    # Preserve the existing arm's mask rule, including exclusion of its last
    # 16 tokens. The changed quota applies across documents, not per document.
    return [i for i, value in enumerate(row['loss_mask'][:-16]) if value]


@torch.no_grad()
def packed_states(target, draft, rows, chunks, budget, cap, explicit_anchors=None):
    device = target.lm_head.weight.device
    ids = torch.zeros((1, budget), dtype=torch.long, device=device)
    docs = torch.full_like(ids, -1)
    positions = torch.zeros_like(ids)
    eligible, offset = [], 0
    for document, row in enumerate(rows):
        length = len(row['input_ids'])
        ids[0, offset:offset + length] = torch.tensor(row['input_ids'], device=device)
        docs[0, offset:offset + length] = document
        positions[0, offset:offset + length] = torch.arange(length, device=device)
        eligible.extend(offset + i for i in eligible_positions(row))
        offset += length
    assert offset <= budget and eligible
    eligible = torch.tensor(eligible, device=device)
    if explicit_anchors is None:
        anchors = torch.sort(eligible[torch.randperm(len(eligible), device=device)[:cap]]).values
    else:
        anchors = torch.tensor(explicit_anchors, dtype=torch.long, device=device)
        assert torch.isin(anchors, eligible).all()
    count = len(anchors)
    index = anchors[:, None] + torch.arange(16, device=device)[None]
    assert torch.equal(docs[0, index], docs[0, anchors, None].expand(-1, 16))
    packed = torch.cat(chunks, 0).to(device)
    aux = torch.zeros((1, budget, 5 * 4096), device=device, dtype=torch.bfloat16)
    aux[0, :offset] = packed[:, :-1].flatten(1)
    last = target.model.norm(packed[:, -1])
    del packed
    full = draft._create_attention_mask(docs, budget, anchors, device) if draft.uses_full_attn else None
    sliding = draft._create_attention_mask(docs, budget, anchors, device,
        sliding_window=draft.sliding_window,
        sliding_window_non_causal=draft.sliding_window_non_causal) if draft.uses_sliding_window_attn else None
    query = torch.full((1, count * 16), draft.mask_token_id, device=device, dtype=torch.long)
    query[:, ::16] = ids[:, anchors]
    noise = draft.embed_tokens(query)
    context = draft.hidden_norm(draft.fc(aux))
    rotary_positions = torch.cat((positions, get_base_indices_for_anchored_blocks(positions[0, anchors], 16)[None]), 1)
    rotary = draft.rotary_emb(aux, rotary_positions)
    del aux
    for i, layer in enumerate(draft.layers):
        noise = layer(hidden_states=noise, target_hidden=context,
            attention_mask=sliding if i in draft.sliding_window_indices else full,
            position_ids=rotary_positions, use_cache=False, position_embeddings=rotary)
    h = draft.norm(noise).reshape(count, 16, -1)[:, 1:].contiguous()
    gold = last[index[:, :-1]].contiguous()
    answers = ids[0, index[:, 1:]].contiguous()
    return h, gold, answers, ids[0, anchors], anchors


def check_first_documents(target, draft, rows, chunks, p, out):
    # Compare packed draft math to the actual old single-document routine on
    # the same online hidden and the same anchors. This is a bounded check.
    rows, chunks = rows[:2], chunks[:2]
    provider = target.hidden_provider
    refs, anchors, offset = [], [], 0
    try:
        target.hidden_provider = lambda row: chunks[next(i for i, item in enumerate(rows) if item is row)]
        for i, row in enumerate(rows):
            seed = p['experiment_seed'] + 900000 + i
            mask = torch.tensor(row['loss_mask'], dtype=torch.bool, device='cuda')[None]
            k = min(8, len(eligible_positions(row)))
            torch.manual_seed(seed)
            local, valid = select_anchors(mask, k, 16)
            assert valid.all()
            anchors.extend((local + offset).tolist())
            torch.manual_seed(seed)
            refs.append(online_states(target, draft, row, k, torch.device('cuda:0')))
            offset += len(row['input_ids'])
        packed = packed_states(target, draft, rows, chunks, offset, len(anchors), anchors)
        reference = [torch.cat([ref[i] for ref in refs], 0) for i in range(4)]
        assert torch.equal(packed[2], reference[2]) and torch.equal(packed[3], reference[3])
        errors = []
        for candidate, expected in zip(packed[:2], reference[:2]):
            diff = candidate.float() - expected.float()
            errors.append(dict(max_abs=float(diff.abs().max()), relative_l2=float(diff.norm() / expected.float().norm().clamp_min(1e-12))))
        assert all(row['relative_l2'] < .02 for row in errors), errors
        dump(out / 'packed_alignment_check.json', dict(passed=True, documents=len(rows), anchors=len(anchors),
            errors=errors, actual_old_online_states_reference=True, answers_and_predecessors_exact=True,
            scope='two training documents, identical anchors and online hidden; not full native parity', utc=now()))
    finally:
        target.hidden_provider = provider


def head_backward(head, basis, target, h, th, answers, anchor, chunk, opt):
    total = 0.
    for start in range(0, len(h), chunk):
        sl = slice(start, start + chunk)
        with torch.no_grad():
            raw = h[sl] @ target.lm_head.weight.T
            gold = th[sl] @ target.lm_head.weight.T
        result = rollout(head, raw, h[sl], target.lm_head.weight, basis,
            anchor[sl], lambda z, j: answers[sl, j])
        assert torch.equal(result.predecessors, torch.cat((anchor[sl, None], answers[sl, :-1]), 1))
        loss = output_loss(result.logits, gold)
        assert torch.isfinite(loss)
        fraction = len(h[sl]) / len(h)
        (loss * fraction).backward()
        opt.accumulate()
        total += float(loss.detach()) * fraction
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    out = Path(p['output']) / 'training'
    out.mkdir(exist_ok=True)
    for field in ['source_files', 'asset_sha256']:
        for path, expected in p[field].items():
            assert sha(path) == expected, path
    assert p['pack_tokens'] == 65536 and p['max_anchors'] == 1536 and p['gradient_accumulation_packs'] == 1
    train = [json.loads(line) for line in Path(p['data_source']).read_text().splitlines()]
    assert len(train) == 20000
    plan = [packs_for_epoch(train, p['experiment_seed'], epoch, p['pack_tokens']) for epoch in range(1, 4)]
    total = sum(map(len, plan))
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(p['training_memory_fraction'], 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target, draft = load_models(torch.device('cuda:0'), f"http://127.0.0.1:{p['port']}/v1", p['transfer_root'], p['hidden_prefetch'])
    assert all(not parameter.requires_grad for model in [target, draft] for parameter in model.parameters())
    provider = target.hidden_provider
    basis = load_file(p['basis_path'])['repair_basis'].to('cuda', dtype=torch.bfloat16)
    output_basis = build_token_basis(target.lm_head.weight, basis)
    input_basis = build_token_basis(target.model.embed_tokens.weight, basis)
    torch.manual_seed(p['experiment_seed'])
    head = DualFeedbackRepair(**p['architecture']).to('cuda', dtype=torch.bfloat16)
    head.bind_features(target.model.embed_tokens.weight, input_basis)
    initial = {name: digest(value) for name, value in head.state_dict().items()}
    original = json.loads(Path(p['baseline_initialization']).read_text())['tensor_sha256']
    assert initial == original, 'Initial head must match the existing 20k arm'
    opt = MasterAdamW(head, list(head.parameters()), [], p)
    steps = 0
    protocol_sha = sha(args.protocol)
    checkpoint = out / 'recovery_checkpoints'
    if not (out / 'initialization.json').exists():
        dump(out / 'initialization.json', dict(fresh_random=True, exact_initial_head_match=True,
            tensor_sha256=initial, reused_identical_frozen_basis=True, target_and_draft_frozen=True, utc=now()))
    dump(out / 'resolved_training.json', dict(packs_per_epoch=list(map(len, plan)), total_updates=total,
        input_tokens_per_pack=p['pack_tokens'], max_anchors_per_pack=p['max_anchors'],
        gradient_accumulation_packs=1, epochs=3, train_count=20000, utc=now()))
    stop_requested = []
    signal.signal(signal.SIGTERM, lambda sig, frame: stop_requested.append(sig))
    signal.signal(signal.SIGINT, lambda sig, frame: stop_requested.append(sig))
    latest = checkpoint / 'resume_latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=True)
        assert state['protocol_sha256'] == protocol_sha and 0 <= state['step'] <= total
        head.load_state_dict(state['head'], strict=True)
        with torch.no_grad():
            for name, parameter, master in opt.pairs:
                master.copy_(state['masters'][name])
                assert torch.equal(parameter.cpu(), master.to(parameter.dtype).cpu())
        opt.optimizer.load_state_dict(state['optimizer'])
        steps = state['step']
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        dump(out / 'resume_loaded.json', dict(step=steps, full_optimizer_and_master_restored=True, utc=now()))
        del state
    else:
        save_checkpoint(checkpoint, head, opt, 0, protocol_sha, 'initial_before_first_update')
    try:
        for epoch, packs in enumerate(plan, 1):
            completed_before = sum(len(item) for item in plan[:epoch - 1])
            skip = max(0, min(len(packs), steps - completed_before))
            if skip == len(packs) and (out / f'epoch_{epoch:02d}.json').exists():
                continue
            provider.begin_epoch([train[i] for indices in packs[skip:] for i in indices])
            for pack_index, indices in enumerate(packs[skip:], skip):
                if stop_requested:
                    raise InterruptedError('user stop at packed optimizer boundary')
                started = time.perf_counter()
                rows = [train[i] for i in indices]
                chunks = [provider(row) for row in rows]
                if steps == 0:
                    check_first_documents(target, draft, rows, chunks, p, out)
                torch.manual_seed(p['experiment_seed'] + steps + 1 + 200000)
                h, th, answers, anchor, positions = packed_states(target, draft, rows, chunks, p['pack_tokens'], p['max_anchors'])
                del chunks
                opt.zero_grad()
                for group in opt.param_groups:
                    group['lr'] = group['base_lr'] * schedule_scale(steps + 1, total, p)
                loss = head_backward(head, output_basis, target, h, th, answers, anchor, p['head_chunk_anchors'], opt)
                norm = float(torch.nn.utils.clip_grad_norm_(opt.master_parameters, p['clip_grad_norm'], error_if_nonfinite=True))
                opt.step()
                opt.zero_grad()
                steps += 1
                elapsed = time.perf_counter() - started
                record = dict(step=steps, total=total, epoch=epoch, pack_index=pack_index,
                    examples=len(indices), input_tokens=sum(len(row['input_ids']) for row in rows),
                    anchors=len(h), loss=loss, grad_norm=norm, seconds=elapsed,
                    anchor_sha256=digest(positions), hidden_requests=provider.requests,
                    hidden_deleted=provider.deleted, utc=now())
                with (out / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                dump(out / 'status.json', dict(status='training', step=steps, total=total, latest=record))
                print(now(), 'TRAIN', steps, '/', total, 'epoch', epoch, 'examples', len(indices),
                    'tokens', record['input_tokens'], 'anchors', len(h), 'loss', loss, 'seconds', round(elapsed, 3), flush=True)
                if steps == 1 or steps % 25 == 0:
                    save_checkpoint(checkpoint, head, opt, steps, protocol_sha, 'periodic')
                del h, th, answers, anchor, positions
            weight = out / f'epoch_{epoch:02d}.safetensors'
            if not weight.exists():
                save_file({key: value.detach().cpu().contiguous().clone() for key, value in head.state_dict().items()}, str(weight))
            save_checkpoint(checkpoint, head, opt, steps, protocol_sha, f'epoch_{epoch}_complete')
            shutil.copy2(checkpoint / 'resume_latest.pt', out / f'epoch_{epoch:02d}_resume.pt')
            dump(out / f'epoch_{epoch:02d}.json', dict(step=steps, weight=str(weight), sha256=sha(weight), full_state=str(out / f'epoch_{epoch:02d}_resume.pt'), utc=now()))
        dump(out / 'summary.json', dict(status='complete', steps=steps, total=total, epochs=3, train_count=20000,
            basis_path=p['basis_path'], basis_sha256=sha(p['basis_path']),
            checkpoints={str(epoch): json.loads((out / f'epoch_{epoch:02d}.json').read_text()) for epoch in range(1, 4)},
            target_and_draft_frozen=True, same_initial_head=True, sampling_unit='whole pack', utc=now()))
        dump(out / 'status.json', dict(status='complete', step=steps, total=total, utc=now()))
    except BaseException as exc:
        save_checkpoint(checkpoint, head, opt, steps, protocol_sha, 'exception')
        dump(out / 'failure.json', dict(step=steps, error=repr(exc), utc=now()))
        raise
    finally:
        provider.close()
        remaining = list(provider.root.iterdir())
        dump(out.parent / 'hidden_cleanup.json', dict(requests=provider.requests, deleted=provider.deleted,
            files_remaining=[str(path) for path in remaining], utc=now()))
        assert provider.requests == provider.deleted and not remaining


if __name__ == '__main__':
    main()
