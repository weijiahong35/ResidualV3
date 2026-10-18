"""Real independent learner; target hidden arrives only through a shared CPU queue."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
CODE=Path(__file__).resolve().parent
RUNTIME=CODE.parent/'runtime_bundle'

def entry(root,arm,in_queue,acks):
    root=Path(root);out=root/arm
    p=json.loads((root/'protocol.json').read_text());p.update(json.loads((root/'assignment.json').read_text()))
    os.environ['CUDA_VISIBLE_DEVICES']=p['learner_uuid'][arm]
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['HEAD_REPAIR_IMPLEMENTATION']=p['repair_implementation']
    os.environ['HEAD_FULL_GATE_PATH']=str(out/'full1536_online_gradient_gate.json')
    os.environ['HEAD_PREVALIDATED_GATE']=p['prevalidated_head_gate']
    os.environ['TIMING_ONLY_NO_PARAMETER_UPDATE']='0'
    os.environ['DRAFT_CHECKPOINT_LAYERS']='4' if arm=='joint' else ''
    os.environ['RECOMPUTE_OUTPUT_ROWS']='1'
    os.environ['TRITON_CACHE_DIR']=str(Path(p['triton_cache_parent'])/f'{arm}_triton')
    os.environ['TORCHINDUCTOR_CACHE_DIR']=str(Path(p['triton_cache_parent'])/f'{arm}_inductor')
    log=(out/'training.log').open('x',buffering=1);sys.stdout=log;sys.stderr=log
    sys.path[:0]=[str(CODE),str(RUNTIME),str(RUNTIME/'src'),str(RUNTIME/'hs_connectors/src')]
    from runtime.utils import dump,now,sha
    import torch
    from safetensors.torch import load_file
    from runtime.live_trainer import load_models
    from runtime.precision import build_token_basis
    from repair.model import DualFeedbackRepair
    from joint_model import MaskEmbedding,TrainablePair,MasterOptim
    try:
        torch.set_num_threads(4);torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        deadline=time.monotonic()+900
        while not (root/'teacher/endpoint.json').exists():
            assert time.monotonic()<deadline;time.sleep(1)
        e=json.loads((root/'teacher/endpoint.json').read_text())
        free,total=torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(free-3*2**30,p['learner_memory_budget_gib']*2**30)/total)
        target,body=load_models(torch.device('cuda'),e['endpoint'],e['root'],16)
        target.hidden_provider.close()
        mask=body.embed_tokens.weight[body.mask_token_id].detach().clone()
        body.embed_tokens=MaskEmbedding(target.model.embed_tokens.weight,body.mask_token_id,mask)
        for name,param in body.named_parameters():
            param.requires_grad_(name.startswith(('layers.','fc.','hidden_norm.','norm.')) or name=='embed_tokens.mask_embedding')
        body.train();torch.manual_seed(p['seed']);head=None;basis=None
        if arm=='joint':
            import joint_features
            dump(out/'selected_softmax_gradient_gate.json',joint_features.gate())
            torch.manual_seed(p['seed'])
            head=DualFeedbackRepair(**json.loads(Path(p['head_config_path']).read_text())).to('cuda',dtype=torch.bfloat16).train()
            b=load_file(p['basis_path'])['repair_basis'].to('cuda',dtype=torch.bfloat16)
            basis=build_token_basis(target.lm_head.weight,b)
            head.bind_features(target.model.embed_tokens.weight,build_token_basis(target.model.embed_tokens.weight,b))
            joint_features.install(head);del b
        pair=TrainablePair(body,head);opt=MasterOptim(pair,p)
        gc.collect();torch.cuda.synchronize()
        dump(out/'model_ready.json',dict(pid=os.getpid(),physical_gpu=p['learner_gpus'][arm],uuid=p['learner_uuid'][arm],body_trainable_parameters=sum(x.numel() for n,x,_ in opt.pairs if n.startswith('body.')),head_trainable_parameters=sum(x.numel() for n,x,_ in opt.pairs if n.startswith('head.')),target_frozen=all(not x.requires_grad for x in target.parameters()),allocated_bytes=torch.cuda.memory_allocated(),utc=now()))
        print(now(),'MODEL_READY',arm,flush=True)
        while not (root/'driver_ready.json').exists():
            assert time.monotonic()<deadline;time.sleep(1)
        ready=json.loads((root/'driver_ready.json').read_text())
        for name,h in ready['source_files'].items():assert sha(name)==h,name
        from production_driver import run_worker
        run_worker(root,arm,p,target,pair,opt,basis,in_queue,acks)
        dump(out/'worker_complete.json',dict(status='complete',utc=now()))
    except BaseException as exc:
        dump(out/'worker_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),utc=now()))
        acks.put(dict(event='failed',arm=arm,error=repr(exc)));raise
    finally:log.close()
