"""Two independent repair heads trained on exactly the same online states."""
import argparse,hashlib,json,time
from pathlib import Path
import openai,torch
from safetensors.torch import load_file,save_file
from .model import DualFeedbackRepair,ARMS,rollout,output_loss
from .protocol import read
from .gradient_audit import audit_gradients
from runtime import live_trainer as old
from runtime.master_optimizer import MasterAdamW
from runtime.live_run import order_for_epoch
from runtime.precision import build_token_basis
from runtime.utils import dump,sha,now

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def hashes(head):return {k:digest(v) for k,v in head.state_dict().items()}
def feature_signature(weight,basis):
    ids=torch.tensor([0,1,42,weight.shape[0]//2,151669,weight.shape[0]-1],device=weight.device)
    return dict(ids=ids.cpu().tolist(),weight_rows_sha256=digest(weight[ids]),basis_rows_sha256=digest(basis[ids]))

class Session:
    def __init__(self,p,out):
        self.p,self.out=p,out;self.train=old.load_examples(p);assert len(self.train)==128
        cfg=p['hidden_servers']['0']
        self.target,self.draft=old.load_models(torch.device('cuda:0'),cfg['endpoint'],cfg['transfer_root'],4)
        self.provider=self.target.hidden_provider;self.provider.client.close()
        self.provider.client=openai.OpenAI(base_url=cfg['endpoint'],api_key='EMPTY',max_retries=0,default_headers={'Connection':'close'})
        self.w=self.target.lm_head.weight
        b=load_file(p['basis_path'])['repair_basis'].to(self.w.device,dtype=torch.bfloat16)
        self.basis=build_token_basis(self.w,b)
        self.weights={'LMHEAD':self.w,'EMBEDDING':self.target.model.embed_tokens.weight}
        self.bases={'LMHEAD':self.basis,'EMBEDDING':build_token_basis(self.weights['EMBEDDING'],b)}
        assert self.weights['LMHEAD'].data_ptr()!=self.weights['EMBEDDING'].data_ptr()
        dump(out/'feature_sources.json',{arm:feature_signature(self.weights[arm],self.bases[arm]) for arm in ARMS})
    def close(self):
        self.provider.close();remaining=[str(f) for f in self.provider.root.iterdir()]
        dump(self.out/'hidden_cleanup.json',dict(requests=self.provider.requests,deleted=self.provider.deleted,files_remaining=remaining))
        assert self.provider.requests==self.provider.deleted and not remaining

def fresh_head(p,*,smoke=False):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['smoke_seed'] if smoke else p['experiment_seed']);head=DualFeedbackRepair(**p['architecture'])
    return head

def fit(s,smoke):
    p,out=s.p,s.out;heads={};optimizers={};initials={};checkpoints={a:{} for a in ARMS}
    for arm in ARMS:
        dest=out/arm;dest.mkdir(exist_ok=False)
        head=fresh_head(p,smoke=smoke).to('cuda:0',dtype=torch.bfloat16)
        head.bind_features(s.weights[arm],s.bases[arm]);heads[arm]=head;initials[arm]=hashes(head)
        dump(dest/'initialization.json',dict(repair_initialization='random_from_scratch',repair_checkpoint_loaded=False,optimizer_state_loaded=False,
            seed=p['smoke_seed'] if smoke else p['experiment_seed'],feature_source=arm,parameter_count=sum(v.numel() for v in head.parameters()),
            parameter_shapes={k:list(v.shape) for k,v in head.named_parameters()},tensor_sha256=initials[arm]))
        if not smoke:
            smoke_init=json.loads((out.parent/'smoke'/arm/'initialization.json').read_text())['tensor_sha256']
            smoke_final=json.loads((out.parent/'smoke'/arm/'final_weight_hashes.json').read_text())
            expected=fresh_head(p).to(dtype=torch.bfloat16)
            assert initials[arm]==hashes(expected) and initials[arm]!=smoke_init and initials[arm]!=smoke_final
            del expected
            dump(dest/'fresh_reset_verified.json',dict(passed=True,matches_fresh_seed=True,differs_from_smoke_initial=True,differs_from_smoke_final=True))
        optimizers[arm]=MasterAdamW(head,list(head.parameters()),[],p)
        assert not optimizers[arm].optimizer.state
    assert initials['LMHEAD']==initials['EMBEDDING']
    dump(out/'initialization_pair.json',dict(passed=True,identical_trainable_initialization=True,independent_optimizers=True))
    if not smoke:dump(out/'fresh_reset_verified.json',dict(passed=True,arms=list(ARMS),identical_trainable_initialization=True))
    steps=0
    for epoch in range(1,(1 if smoke else p['epochs'])+1):
        indices=order_for_epoch(128,p['experiment_seed'],epoch)
        if smoke:indices=indices[:2]
        s.provider.begin_epoch([s.train[i] for i in indices])
        for index in indices:
            started=time.monotonic();steps+=1;torch.manual_seed(p['experiment_seed']+steps+200000)
            h,th,answers,anchor=old.online_states(s.target,s.draft,s.train[index],p['max_anchors'],torch.device('cuda:0'))
            count=len(h);assert 0<count<=p['max_anchors'];loss_sum={a:0. for a in ARMS}
            for arm in ARMS:
                heads[arm].train();optimizers[arm].zero_grad()
                for group in optimizers[arm].param_groups:group['lr']=group['base_lr']*old.schedule_scale(steps,384,p)
            for start in range(0,count,p['head_chunk_anchors']):
                sl=slice(start,start+p['head_chunk_anchors'])
                with torch.no_grad():raw=h[sl]@s.w.T;gold=th[sl]@s.w.T
                for arm in ARMS:
                    head= heads[arm];opt=optimizers[arm]
                    if steps==1 and start==0:
                        from .validate import real_checks
                        evidence,gradients=real_checks(head,raw[:1],h[sl][:1],s.w,s.basis,anchor[sl][:1],lambda z,j:answers[sl][:1,j])
                        dump(out/arm/'real_input_effects.json',evidence);dump(out/arm/'real_input_gradients.json',gradients)
                    r=rollout(head,raw,h[sl],s.w,s.basis,anchor[sl],lambda z,j:answers[sl,j])
                    assert torch.equal(r.predecessors[:,0],anchor[sl]) and torch.equal(r.predecessors[:,1:],answers[sl,:-1])
                    loss=output_loss(r.logits,gold);assert torch.isfinite(loss)
                    (loss*(len(h[sl])/count)).backward();opt.accumulate();loss_sum[arm]+=float(loss.detach())*len(h[sl])/count
                    del r,loss
                del raw,gold
            common=dict(step=steps,epoch=epoch,index=index,anchors=count,anchor_sha256=digest(anchor),answer_sha256=digest(answers),
                hidden_requests=s.provider.requests,hidden_deleted=s.provider.deleted,utc=now())
            latest={}
            for arm in ARMS:
                head=heads[arm];opt=optimizers[arm]
                grads={name:float(master.grad.abs().sum()) for name,_,master in opt.pairs if master.grad is not None}
                fusion=next(master for name,_,master in opt.pairs if name=='fusion.1.weight')
                slices={k:float(fusion.grad[:,i*head.width:(i+1)*head.width].abs().sum()) for i,k in enumerate(('hidden','hard_previous','q','soft_previous'))}
                gradient_audit=audit_gradients(grads,dict(head.named_parameters()),slices)
                norm=float(torch.nn.utils.clip_grad_norm_(opt.master_parameters,p['clip_grad_norm'],error_if_nonfinite=True));opt.step();opt.zero_grad()
                row=dict(**common,arm=arm,loss=loss_sum[arm],grad_norm=norm,gradient_sums=grads,fusion_gradient_sums=slices,**gradient_audit,seconds=time.monotonic()-started)
                with (out/arm/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                dump(out/arm/'status.json',dict(status='training',step=steps,total=2 if smoke else 384,latest=row));latest[arm]=row
                print(now(),arm,'step',steps,'anchors',count,'loss',loss_sum[arm],'seconds',round(row['seconds'],2),flush=True)
            del h,th,answers,anchor
            dump(out/'status.json',dict(status='training',step=steps,total=2 if smoke else 384,latest=latest))
        if not smoke:
            for arm in ARMS:
                path=out/arm/f'epoch_{epoch:02d}.safetensors'
                save_file({k:v.detach().cpu().contiguous().clone() for k,v in heads[arm].state_dict().items()},str(path))
                checkpoints[arm][str(epoch)]=dict(path=str(path),sha256=sha(path))
    for arm in ARMS:
        final=hashes(heads[arm]);assert final!=initials[arm]
        dump(out/arm/'final_weight_hashes.json',final)
        dump(out/arm/'status.json',dict(status='complete',step=steps,total=2 if smoke else 384,utc=now()))
        dump(out/arm/'summary.json',dict(status='complete',steps=steps,checkpoints=checkpoints[arm],smoke=smoke,repair_initialization='random_from_scratch'))
    dump(out/'status.json',dict(status='complete',step=steps,total=2 if smoke else 384,utc=now()))
    dump(out/'summary.json',dict(status='complete',steps=steps,arms=list(ARMS),checkpoints=checkpoints,smoke=smoke,repair_initialization='random_from_scratch',utc=now()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    p=read(a.protocol);a.output.mkdir(exist_ok=False);s=None
    torch.set_num_threads(4);torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(p['training_memory_fraction'],0)
    try:s=Session(p,a.output);fit(s,a.smoke)
    except BaseException as exc:dump(a.output/'status.json',dict(status='failed',error=repr(exc),utc=now()));raise
    finally:
        if s is not None:s.close()
if __name__=='__main__':main()
