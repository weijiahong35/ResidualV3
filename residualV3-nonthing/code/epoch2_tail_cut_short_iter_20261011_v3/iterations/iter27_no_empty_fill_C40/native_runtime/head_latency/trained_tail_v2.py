"""Frozen epoch2 readouts on repaired q, with causal one-position delayed gating."""
import os
import torch
from torch import nn
import triton
import triton.language as tl
from .q45_features import features

BRANCHES = ('Q45_MLP', 'Q45_REPAIR_LATENT_MLP')

def prepare(head, config):
    from .tail_pipeline import prepare as prepare_streams
    prepare_streams(head, config)
    state = torch.load(os.environ['TAIL_AUX_CHECKPOINT'], map_location='cpu', weights_only=True)
    assert state['format'] == 'two_detached_online_MLPs_v1' and state['initialized']
    device = head.direction_head.weight.device
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        nets = nn.ModuleDict({name: nn.Sequential(nn.Linear(d,32), nn.SiLU(), nn.Linear(32,2))
                              for name,d in zip(BRANCHES,(45,301))})
    for name, net in nets.items():
        net.load_state_dict({k.removeprefix(name+'.net.'):v for k,v in state['model'].items()
                             if k.startswith(name+'.net.')})
    # Frozen readouts are deliberately outside the repair parameter namespace.
    # Same BF16 operands as autocast, cast once outside the captured15-step
    # loop instead of recasting the frozenFP32 weights at each position.
    head.__dict__['_tail_readouts'] = nets.to(device,dtype=torch.bfloat16).eval().requires_grad_(False)
    head._tail_mean = state['mean'].to(device)
    head._tail_scale = state['scale'].to(device)
    head._tail_thresholds = torch.tensor(config['thresholds'], device=device, dtype=torch.float32)
    head._tail_branch = os.environ.get('TAIL_AUX_BRANCH', 'Q45_MLP')
    # vLLM's internal warmup synthesizes full15 verification lists directly,
    # bypassing DraftTokensHandler. A mutable graph input fills all q slots
    # only for those internal requests; normal decode leaves it zero.
    head._tail_sampling_override=torch.zeros((),device=device,dtype=torch.int32)
    head._tail_sampling_override_host=False
    assert head._tail_branch in (*BRANCHES, 'BOTH')
    from .tail_graph_state import allocate_tail_state
    allocate_tail_state(head, head._tail_events, 4 if head._tail_branch == 'BOTH' else 2)

def predict(head, logits, latent, active=None, *, known_all_active=False):
    if active is None or os.environ.get('TAIL_FAST_FEATURES','1')=='0' or (known_all_active and logits.shape[0]>=16):
        q45, _ = features(logits)
    else:
        from .tail_fast import features as gated_features
        q45 = gated_features(logits,active)
    chosen = BRANCHES if head._tail_branch == 'BOTH' else (head._tail_branch,)
    outputs = []
    for name in chosen:
        if name=='Q45_MLP':
            normalized=(q45-head._tail_mean[:45])/head._tail_scale[:45]
        else:
            x=torch.cat((q45,latent.detach().float()),-1)
            normalized=(x-head._tail_mean)/head._tail_scale
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = head._tail_readouts[name](normalized)
        outputs.append(output.float().sigmoid())
    return torch.cat(outputs,-1), q45[:,0]

@triton.jit
def _finish(PROB, ACTIVE, SURV, PATIENCE, MAXQ, SCORES, MAXIMUM, THRESH, TEMP, MAP,
            N:tl.constexpr,L:tl.constexpr,P:tl.constexpr,J:tl.constexpr,MS:tl.constexpr,
            MINIMUM:tl.constexpr,PAT:tl.constexpr,FORCE:tl.constexpr,B:tl.constexpr):
    rows=tl.arange(0,B)
    alive=tl.load(ACTIVE+rows*(L+2)+J,rows<N,other=0)
    prior=tl.load(SURV+rows*(L+1)+J,rows<N,other=1.)
    old=tl.load(PATIENCE+rows*(L+1)+J,rows<N,other=0)
    mappings=tl.load(MAP+rows,rows<N,other=0)
    temperature=tl.load(TEMP+mappings)
    t=tl.where(temperature==0.,0,1)
    confidence=tl.load(PROB+rows*P+t,rows<N,other=1.)
    survival=tl.where(alive!=0,prior*confidence,prior)
    threshold=tl.load(THRESH+t)
    low=(survival<threshold)&(J+1>=MINIMUM)
    count=tl.where(low,old+1,0)
    next_alive=tl.load(ACTIVE+rows*(L+2)+J+1,rows<N,other=0)
    keep=next_alive&tl.cast(count<PAT,tl.int32)
    if FORCE>0:
        keep=tl.full((B,),int(J+2<FORCE),tl.int32)
    tl.store(SURV+rows*(L+1)+J+1,survival,rows<N)
    tl.store(PATIENCE+rows*(L+1)+J+1,count,rows<N)
    tl.store(MAXQ+rows*L+J,tl.load(MAXIMUM+rows*MS,rows<N,other=0.),rows<N)
    for k in range(P):
        value=tl.load(PROB+rows*P+k,rows<N,other=1.)
        tl.store(SCORES+(rows*L+J)*P+k,value,rows<N)
    tl.store(ACTIVE+rows*(L+2)+J+2,keep,rows<N)

def signal(head,z,latent,active,survival,patience,maxq,scores,temp,mappings,j,force):
    # With minimum8, patience2 and one-position delay, the first10 positions
    # cannot have stopped. StockTopK is faster for a fully active large batch.
    config=head._tail_config
    earliest=config['minimum_repair']+config['patience']
    probabilities, maximum = predict(head,z,latent,active[:,j],known_all_active=j<earliest and force==0)
    n=z.shape[0]; length=maxq.shape[1]
    _finish[(1,)](probabilities,active,survival,patience,maxq,scores,maximum,
        head._tail_thresholds,temp,mappings,n,length,probabilities.shape[-1],j,maximum.stride(0),
        config['minimum_repair'],config['patience'],force,triton.next_power_of_2(n),num_warps=4)
