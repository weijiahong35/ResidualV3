"""Bounded real online training updates, isolated from completed experiment arms."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
import urllib.request

import torch
from safetensors.torch import load_file, save_file

CODE = Path(__file__).resolve().parent


def run(output, source_run, service, library=None, allocator='native'):
    from arena import stats
    from runtime.utils import dump, now, sha
    from runtime.live_trainer import load_models
    from runtime.precision import build_token_basis
    from runtime.master_optimizer import MasterAdamW
    from repair.model import DualFeedbackRepair
    from repair.run import digest
    from prepare import install as install_prepare
    from dp_teacher import install
    from pipeline import PackPipeline
    import runtime.vllm_hidden as hidden
    import train as learner

    output = Path(output); output.mkdir(exist_ok=False)
    source_run, service = Path(source_run), Path(service)
    protocol = json.loads((source_run / 'training_arms/CE_DPACE_OUTPUT/training_protocol.json').read_text())
    endpoint = json.loads((service / 'endpoint.json').read_text())
    checkpoint = source_run / 'training_arms/CE_DPACE_OUTPUT/training/epoch_03_resume.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['step'] == 894
    source = [json.loads(line) for line in Path(protocol['data_source']).read_text().splitlines()]
    packs = learner.packs_for_epoch(source, protocol['experiment_seed'], 2, 65536)[:12]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The private allocator pilot uses only the default stream, without CUDA graphs.
    from torch._inductor import config as inductor_config
    inductor_config.triton.cudagraphs = False
    if library is None:
        torch.cuda.set_per_process_memory_fraction(68 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        if os.environ.get('STUDY_WORKSPACE_READY') != '1':
            workspace = torch.empty(67 * 2**30, dtype=torch.uint8, device='cuda')
            del workspace
    os.environ['THROUGHPUT_REQUESTS'] = '32'
    install(output / 'initial_hidden.jsonl', dict(endpoint=endpoint['endpoint'], root=endpoint['root'], dp_size=endpoint['dp_size']))
    dump(output / 'status.json', dict(status='loading_models', pid=os.getpid(), utc=now()))
    target, draft = load_models(torch.device('cuda:0'), endpoint['endpoint'], endpoint['root'], 16)
    target.hidden_provider.close()
    basis = load_file(protocol['basis_path'])['repair_basis'].to('cuda', dtype=torch.bfloat16)
    output_basis = build_token_basis(target.lm_head.weight, basis)
    input_basis = build_token_basis(target.model.embed_tokens.weight, basis)
    learner.ACTIVE_ARM = 'CE_DPACE_OUTPUT'; learner.ACTIVE_MODE = 'sequential'
    summary = []
    anchor_hashes = {}
    if os.environ.get('DO_FULL_HEAD_GATE') == '1':
        from measure_head import run as measure
        os.environ['THROUGHPUT_REQUESTS'] = '128'
        provider = hidden.OnlineHiddenProvider(endpoint['endpoint'],endpoint['root'],protocol['target_model'],prefetch=16)
        pipeline = None
        try:
            pipeline = PackPipeline(provider,source,packs[:1],target=target,draft=draft)
            indices,rows,chunk = pipeline.get()
            torch.manual_seed(protocol['experiment_seed'] + 200000 + 299)
            h,th,answers,anchor,_ = learner.packed_states(target,draft,rows,chunk,65536,1536,shard_anchors=512)
            torch.cuda.synchronize();chunk.release();del chunk
            pipeline.release();pipeline.close();pipeline=None
            head=DualFeedbackRepair(**protocol['architecture']).to('cuda',dtype=torch.bfloat16)
            head.load_state_dict(state['head'],strict=True);head.bind_features(target.model.embed_tokens.weight,input_basis)
            install_prepare(head)
            gate=output/'full1536_head_probe';gate.mkdir()
            measure(gate,head,target.lm_head.weight,output_basis,h,th,answers,anchor,dump,now)
            del head,h,th,answers,anchor
            gc.collect()
        finally:
            if pipeline is not None:pipeline.close()
            provider.close()
            dump(output/'full1536_gate_hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted))
            assert provider.requests==provider.deleted
        del provider,pipeline
        gc.collect()
    modes=['sequential','parallel_tf','parallel_position_bmm','sequential']
    for phase_index, mode in enumerate(modes,1):
        concurrency=128
        learner.ACTIVE_MODE=mode
        phase = output / f'phase_{phase_index:02d}_{mode}_requests{concurrency}'; phase.mkdir()
        reset_ok = False
        try:
            url = endpoint['endpoint'].removesuffix('/v1') + '/reset_prefix_cache'
            request = urllib.request.Request(url, data=b'', method='POST')
            with urllib.request.urlopen(request, timeout=30) as response:
                reset_ok = response.status == 200
        except Exception as exc:
            dump(phase / 'prefix_cache_reset.json', dict(success=False, error=repr(exc)))
        os.environ['THROUGHPUT_REQUESTS'] = str(concurrency)
        install(phase / 'hidden_requests.jsonl', dict(endpoint=endpoint['endpoint'], root=endpoint['root'], dp_size=endpoint['dp_size']))
        provider = hidden.OnlineHiddenProvider(endpoint['endpoint'], endpoint['root'], protocol['target_model'], prefetch=16)
        target.hidden_provider = provider
        head = DualFeedbackRepair(**protocol['architecture']).to('cuda', dtype=torch.bfloat16)
        head.load_state_dict(state['head'], strict=True)
        head.bind_features(target.model.embed_tokens.weight, input_basis)
        install_prepare(head)
        opt = MasterAdamW(head, list(head.parameters()), [], protocol)
        with torch.no_grad():
            for name, parameter, master in opt.pairs:
                master.copy_(state['masters'][name])
                assert torch.equal(parameter.cpu(), master.to(parameter.dtype).cpu())
        opt.optimizer.load_state_dict(state['optimizer'])
        for group in opt.optimizer.param_groups:
            group['lr'] = protocol['stage1_lr'] * protocol['min_lr_fraction']
        pipeline = None
        records = []
        try:
            pipeline = PackPipeline(provider, source, packs, target=target, draft=draft)
            for pack_index, indices in enumerate(packs):
                started = time.perf_counter()
                returned, rows, chunk = pipeline.get()
                wait = time.perf_counter() - started
                assert returned == indices
                torch.manual_seed(protocol['experiment_seed'] + 200000 + 299 + pack_index)
                begin = time.perf_counter()
                h, th, answers, anchor, positions = learner.packed_states(target, draft, rows, chunk, 65536, 1536, shard_anchors=512)
                torch.cuda.synchronize(); body = time.perf_counter() - begin
                assert len(h) == 1536
                key = digest(positions)
                assert key == anchor_hashes.setdefault(pack_index, key), 'Anchor order changed between profiles'
                chunk.release(); del chunk
                with pipeline.stage_lock:
                    begin = time.perf_counter(); opt.zero_grad()
                    loss = learner.head_backward(head, output_basis, target, h, th, answers, anchor, 1536, opt)
                    torch.cuda.synchronize(); head_seconds = time.perf_counter() - begin
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(opt.master_parameters, 1., error_if_nonfinite=True))
                    if library:
                        assert all(library.arena_owns(master.grad.data_ptr()) for master in opt.master_parameters if master.grad is not None)
                    opt.step(); opt.zero_grad()
                torch.cuda.synchronize(); pipeline.release()
                elapsed = time.perf_counter() - started
                record = dict(phase=phase_index, head_mode=mode, teacher_dp_size=endpoint['dp_size'], request_concurrency=concurrency, pack=pack_index + 1,
                    measured=pack_index >= 3, input_tokens=sum(len(row['input_ids']) for row in rows),
                    examples=len(rows), anchors=len(h), seconds=elapsed, hidden_wait_seconds=wait,
                    draft_seconds=body, head_seconds=head_seconds, loss=loss, grad_norm=grad_norm,
                    anchor_sha256=key, peak_running_requests=provider.ordered.peak_running,
                    peak_pending_results=provider.ordered.peak_pending,
                    arena=stats(library) if library else None, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(), utc=now())
                records.append(record)
                with (phase / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record) + '\n')
                dump(output / 'status.json', dict(status='benchmarking', allocator=allocator, pid=os.getpid(), latest=record, utc=now()))
                print(now(), 'BENCH', allocator, 'phase', phase_index, 'requests', concurrency,
                    'mode',mode,'DP',endpoint['dp_size'],'pack', pack_index + 1, '/12', 'seconds', round(elapsed, 3), 'wait', round(wait, 3),
                    'draft', round(body, 3), 'head', round(head_seconds, 3), flush=True)
                del h, th, answers, anchor, positions
            pipeline.close(); pipeline = None
            save_file({name: value.detach().cpu().contiguous().clone() for name, value in head.state_dict().items()}, str(phase / 'pilot_head.safetensors'))
            torch.save(dict(head={name: value.detach().cpu() for name,value in head.state_dict().items()},
                            masters={name: master.detach().cpu() for name,_,master in opt.pairs},
                            optimizer=opt.optimizer.state_dict(), cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                            updates=12, source_step=894), phase / 'pilot_full_state.pt')
        finally:
            if pipeline is not None: pipeline.close()
            provider.close()
            dump(phase / 'hidden_cleanup.json',dict(requests=provider.requests,deleted=provider.deleted,
                 no_persistent_hidden=True,utc=now()))
            assert provider.requests == provider.deleted
        measured = [row for row in records if row['measured']]
        seconds = sum(row['seconds'] for row in measured)
        result = dict(phase=phase_index, head_mode=mode, teacher_dp_size=endpoint['dp_size'], request_concurrency=concurrency, allocator=allocator, measured_packs=9,
            mean_seconds=statistics.mean(row['seconds'] for row in measured),
            mean_hidden_wait_seconds=statistics.mean(row['hidden_wait_seconds'] for row in measured),
            mean_draft_seconds=statistics.mean(row['draft_seconds'] for row in measured),
            mean_head_seconds=statistics.mean(row['head_seconds'] for row in measured),
            input_tokens_per_second=sum(row['input_tokens'] for row in measured) / seconds,
            anchors_per_second=1536 * len(measured) / seconds, prefix_cache_reset=reset_ok, unique_request_cache_salt=True,
            source_checkpoint_sha256=sha(checkpoint))
        summary.append(result); dump(phase / 'summary.json', result)
        dump(output / 'summary.json', dict(status='running', profiles=summary, utc=now()))
        del head, opt, provider; gc.collect()
    dump(output / 'summary.json', dict(status='complete', profiles=summary, source_step=894,
         trainable='repair head only; frozen target/draft', sampling_unchanged=True,
         mathematical_gate='same online states and three objectives compared before this run',
         scope='48 isolated pilot updates per teacher configuration; not long training or native inference performance', utc=now()))
    dump(output / 'status.json', dict(status='complete', utc=now()))
