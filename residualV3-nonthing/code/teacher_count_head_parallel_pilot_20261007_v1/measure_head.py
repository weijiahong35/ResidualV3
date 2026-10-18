"""Online full1536 TF loss/gradient comparisons and fixed-state head timing; no persistent cache."""
import gc
import time
import torch
from repair.model import rollout
from losses import objective
from parallel_modes import parallel_logits

MODES = ['sequential', 'parallel_tf', 'parallel_position_bmm']
ARMS = ['OUTPUT', 'CE_DPACE', 'CE_DPACE_OUTPUT']


def logits_for(mode, head, raw, h, weight, basis, anchor, answers):
    if mode == 'sequential':
        return rollout(head, raw, h, weight, basis, anchor, lambda z, j: answers[:, j]).logits
    return parallel_logits(head, raw, h, basis, anchor, answers, mode)


def gradient_vector(head):
    return torch.cat([p.grad.detach().float().flatten() if p.grad is not None else
                      torch.zeros_like(p, dtype=torch.float32).flatten() for p in head.parameters()])


def run(output, head, weight, basis, h, th, answers, anchor, dump, now):
    assert len(h) == 1536 and h.shape[1] == 15 and not head.use_soft
    with torch.no_grad():
        raw, gold = h @ weight.T, th @ weight.T
    rows, timings = [], []
    for arm in ARMS:
        reference = None
        for mode in MODES:
            head.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            z = logits_for(mode, head, raw, h, weight, basis, anchor, answers)
            loss = objective(z, gold, arm)
            prediction = z.detach().argmax(-1)
            loss.backward()
            torch.cuda.synchronize()
            grad = gradient_vector(head)
            assert torch.isfinite(loss) and torch.isfinite(grad).all()
            if reference is None:
                reference = (float(loss), prediction.clone(), grad.clone())
            ref_loss, ref_pred, ref_grad = reference
            rel = float((grad-ref_grad).norm()/ref_grad.norm().clamp_min(1e-30))
            cosine = float(torch.nn.functional.cosine_similarity(grad[None], ref_grad[None]))
            agreement = float((prediction==ref_pred).float().mean())
            row = dict(arm=arm, mode=mode, anchors=1536, positions=15, loss=float(loss),
                       loss_abs_difference=abs(float(loss)-ref_loss), gradient_relative_l2=rel,
                       gradient_cosine=cosine, argmax_agreement=agreement,
                       logits_argmax_mismatches=int((prediction!=ref_pred).sum()),
                       gradient_bitwise_equal=torch.equal(grad,ref_grad),
                       loss_bitwise_equal=float(loss)==ref_loss,
                       within_small_error_tolerance=rel<=0.002 and cosine>=0.99999 and agreement==1.,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),utc=now())
            rows.append(row)
            print(now(), 'FULL1536_GATE', row, flush=True)
            dump(output/'numerical_progress.json',dict(checks=rows,production_replacement=False,utc=now()))
            del z,loss,prediction,grad
        del reference
    head.zero_grad(set_to_none=True)
    gc.collect()
    for mode in MODES:
        samples=[]
        for iteration in range(8):
            head.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            started=time.perf_counter();begin.record()
            z=logits_for(mode,head,raw,h,weight,basis,anchor,answers)
            loss=objective(z,gold,'CE_DPACE_OUTPUT')
            middle.record();loss.backward();end.record();end.synchronize()
            sample=dict(iteration=iteration,measured=iteration>=3,
                        wall_seconds=time.perf_counter()-started,
                        forward_and_loss_cuda_seconds=begin.elapsed_time(middle)/1000,
                        backward_cuda_seconds=middle.elapsed_time(end)/1000,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            samples.append(sample);del z,loss
        measured=[x for x in samples if x['measured']]
        record=dict(mode=mode,anchors=1536,measured_iterations=len(measured),
                    mean_wall_seconds=sum(x['wall_seconds'] for x in measured)/len(measured),
                    mean_forward_loss_cuda_seconds=sum(x['forward_and_loss_cuda_seconds'] for x in measured)/len(measured),
                    mean_backward_cuda_seconds=sum(x['backward_cuda_seconds'] for x in measured)/len(measured),
                    max_allocated_bytes=max(x['peak_allocated_bytes'] for x in samples),samples=samples,
                    scope='Fixed online states in transient GPU memory; target projection excluded; head features/network/scatter/fused loss/backward included; no Adam updates.')
        timings.append(record);print(now(),'HEAD_FIXED_STATE_TIMING',mode,record['mean_wall_seconds'],flush=True)
    head.zero_grad(set_to_none=True)
    dump(output/'summary.json',dict(status='complete',checks=rows,timings=timings,
         weights_unchanged=True,no_optimizer_updates=True,production_replacement=False,
         dtype='BF16 weights/features/logits; existing FP32 reductions',
         native_recursion_unchanged=True,scope='One online full1536 pack, nine full-sized gradient comparisons and three head microbenchmarks.',utc=now()))
    del raw,gold
    gc.collect()
