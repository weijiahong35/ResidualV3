"""Same real pack, same weights/anchors: main-gradient parity and paired time."""
import copy
import gc
import importlib.util
from pathlib import Path
import statistics
import time
import traceback
import torch
from full_checkpoint import atomic,digest_tensor
from joint_runner import update
from runtime.utils import now


def _run(out,p,pair,target,opt,basis,pack,seed):
    saved=copy.deepcopy(opt.auxiliary.state_dict())
    reference_path=Path(p['auxiliary_predictor']['original_joint_runner'])
    spec=importlib.util.spec_from_file_location('original_q45_gate_runner',reference_path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    captured={}
    def capture(optimizer):
        captured.update({name:digest_tensor(master.grad) for name,_,master in optimizer.pairs
                         if master.grad is not None})
    opt.probe_capture=capture
    opt.auxiliary_enabled=False
    before={name:digest_tensor(value) for name,value,_ in opt.pairs}
    baseline=original.update(pair,target,basis,pack,seed,1536,opt,check=True,step=False)
    hashes=dict(captured);rng=torch.cuda.get_rng_state().clone();captured.clear()
    gc.collect()
    opt.auxiliary_enabled=True
    opt.auxiliary.check_features=True
    augmented=update(pair,target,basis,pack,seed,1536,opt,check=True,step=False)
    equal=hashes==captured and baseline['loss']==augmented['loss']
    unchanged=all(digest_tensor(value)==before[name] for name,value,_ in opt.pairs)
    rng_equal=torch.equal(rng,torch.cuda.get_rng_state())
    assert unchanged,'A no-step gate changed main parameters'
    opt.probe_capture=None
    gradient=dict(main_loss_exact=baseline['loss']==augmented['loss'],
        main_parameter_gradients_bitwise_equal=hashes==captured,
        parameters_checked=len(hashes),main_weights_bitwise_unchanged=unchanged,
        main_cuda_rng_bitwise_equal=rng_equal,
        actual_repaired_q45_feature_max_abs=opt.auxiliary.actual_feature_max_abs)
    records=[]
    for i in range(6):
        for active in ([False,True] if i%2==0 else [True,False]):
            opt.auxiliary_enabled=active
            gc.collect()
            torch.cuda.synchronize();start=time.perf_counter()
            values=update(pair,target,basis,pack,seed,1536,opt,step=False)
            torch.cuda.synchronize();wall=time.perf_counter()-start
            records.append(dict(pair=i,measured=i>=2,auxiliary=active,compute_wall_seconds=wall,
                auxiliary_seconds=values.get('auxiliary_seconds',0.),loss=values['loss'],
                anchors=values['anchors'],anchor_sha256=values['anchor_sha256'],
                peak_allocated_bytes=values['peak_allocated_bytes'],
                stages={k:v for k,v in values.items() if k.endswith('_wall')}))
            print(now(),'AUX_Q45_AB',records[-1],flush=True)
    measured=[x for x in records if x['measured']]
    baseline_mean=statistics.mean(x['compute_wall_seconds'] for x in measured if not x['auxiliary'])
    augmented_mean=statistics.mean(x['compute_wall_seconds'] for x in measured if x['auxiliary'])
    auxiliary_mean=statistics.mean(x['auxiliary_seconds'] for x in measured if x['auxiliary'])
    overhead=(augmented_mean-baseline_mean)/baseline_mean
    direct=auxiliary_mean/baseline_mean
    passed=equal and rng_equal and overhead<=p['auxiliary_predictor']['max_compute_overhead_fraction'] and direct<=p['auxiliary_predictor']['max_compute_overhead_fraction']
    opt.auxiliary.load_state_dict(saved);opt.auxiliary_enabled=passed;opt.zero_grad()
    opt.auxiliary_gate_complete=True
    result=dict(status='passed' if passed else 'disabled',gradient_gate=gradient,
        baseline_compute_seconds=baseline_mean,augmented_compute_seconds=augmented_mean,
        compute_overhead_fraction=overhead,direct_auxiliary_seconds=auxiliary_mean,
        direct_auxiliary_fraction=direct,predictor_parameters=11268,
        branch_parameter_counts=dict(Q45_MLP=1538,Q45_REPAIR_LATENT_MLP=9730),head_chunk=1536,
        acceptance_threshold_fraction=p['auxiliary_predictor']['max_compute_overhead_fraction'],
        records=records,live_main_optimizer_steps_during_gate=0,
        original_main_checkpoint_and_RNG_preserved=True,
        continuation='enable auxiliary' if passed else 'continue original main training with auxiliary disabled',
        auxiliary_probe_updates_discarded=True,scope='real online pack, same weights and anchors; no native AL/TPS',utc=now())
    atomic(out/'auxiliary_live_gate.json',result)
    return result


def run(out,p,pair,target,opt,basis,pack,seed):
    state=copy.deepcopy(opt.auxiliary.state_dict())
    try:
        return _run(out,p,pair,target,opt,basis,pack,seed)
    except torch.OutOfMemoryError as exc:
        error=str(exc);frames=traceback.format_exc()
        traceback.clear_frames(exc.__traceback__);exc.__traceback__=None
        opt.probe_capture=None;opt.zero_grad();opt.auxiliary.pending=[]
        opt.auxiliary.load_state_dict(state);opt.auxiliary_enabled=False
        opt.auxiliary_gate_complete=True
        gc.collect();torch.cuda.empty_cache()
        result=dict(status='disabled',reason='auxiliary live gate OOM; no main optimizer step',
            error=error,traceback=frames,compute_overhead_fraction=None,
            gradient_gate=dict(main_parameter_gradients_bitwise_equal=False,not_certified=True),
            continuation='original main training without predictor',utc=now())
        atomic(out/'auxiliary_live_gate.json',result)
        return result
