"""Prompt-disjoint, RAM-only classification and repair-coverage diagnostics."""
import json
from pathlib import Path
import numpy as np
import torch
from runtime.utils import dump, now


def metrics(y, score, weights=None):
    y=np.asarray(y,dtype=bool);score=np.asarray(score,dtype=np.float64)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    original_states=len(y)
    active=w>0;y=y[active];score=score[active];w=w[active]
    positive=float(w[y].sum());negative=float(w[~y].sum())
    if positive==0 or negative==0:
        return dict(AUROC=None,AP=None,positive_rate=positive/max(positive+negative,1.),states=original_states)
    order=np.argsort(-score,kind='stable');y=y[order];w=w[order];z=score[order]
    ends=np.r_[np.flatnonzero(z[1:]!=z[:-1]),len(z)-1]
    tp=np.r_[0.,np.cumsum(w*y)[ends]];fp=np.r_[0.,np.cumsum(w*(~y))[ends]]
    recall=tp/positive;fpr=fp/negative;precision=tp[1:]/(tp[1:]+fp[1:])
    auc=float((np.diff(fpr)*(recall[1:]+recall[:-1])/2).sum());ap=float((np.diff(recall)*precision).sum())
    return dict(AUROC=auc,AP=ap,positive_rate=positive/(positive+negative),states=original_states)


@torch.inference_mode(False)
@torch.enable_grad()
def fit(x,y,train,kind):
    torch.set_num_threads(2)
    mu=x[train].mean(0);scale=x[train].std(0).clip(.01)
    values=torch.from_numpy(((x-mu)/scale).astype(np.float32))
    target=torch.from_numpy(y.astype(np.float32))
    ids=torch.from_numpy(np.flatnonzero(train))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261009)
        model=torch.nn.Linear(x.shape[1],1) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.SiLU(),torch.nn.Linear(32,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.015,weight_decay=.01)
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(values[ids]).flatten(),target[ids])
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():score=model(values).flatten().sigmoid().numpy().copy()
    parameters=dict(mean=mu.tolist(),scale=scale.tolist(),kind=kind,
        weights={k:v.detach().tolist() for k,v in model.state_dict().items()},parameter_count=sum(v.numel() for v in model.parameters()))
    return score,parameters


def coverage(score,threshold,subset,meta,positive_gain):
    take=subset & (score<threshold)
    denom=float(positive_gain[subset].sum())
    utility=np.asarray([g>0 if v['T']==0 else g>.005 for g,v in zip(positive_gain,meta)])
    grouped={}
    for i in np.flatnonzero(subset):
        key=(meta[i]['prompt'],meta[i]['T'],meta[i]['block'])
        grouped.setdefault(key,[]).append(int(i))
    block_ids=[ids for ids in grouped.values() if len(ids)==15]
    block_skip=[ids for ids in block_ids if bool(np.all(score[ids]<threshold))]
    block_lost=sum(float(positive_gain[ids].sum()) for ids in block_skip)
    reached=np.array([v['reached'] for v in meta],dtype=bool)
    reached_skip=take & reached
    reached_total=subset & reached
    mismatch=np.asarray([v['argmax_mismatch'] if v['T']==0 else v['tv_raw']>=.10 for v in meta])
    tv=np.asarray([v['tv_raw'] for v in meta])
    return dict(threshold=float(threshold),state_skip_fraction=float(take.sum()/max(subset.sum(),1)),
        skipped_mismatch_rate=float(mismatch[take].mean()) if take.any() else None,
        skipped_mean_TV=float(tv[take].mean()) if take.any() else None,
        beneficial_state_false_skip=float((take&utility).sum()/max((subset&utility).sum(),1)),
        positive_utility_proxy_lost_fraction=float(positive_gain[take].sum()/denom) if denom>0 else None,
        reached_state_skip_fraction=float(reached_skip.sum()/max(reached_total.sum(),1)),
        reached_positive_utility_proxy_lost_fraction=float(positive_gain[reached_skip].sum()/max(positive_gain[reached_total].sum(),1e-30)),
        full_blocks=len(block_ids),whole_block_skip_fraction=len(block_skip)/max(len(block_ids),1),
        whole_block_positive_utility_proxy_lost_fraction=block_lost/denom if denom>0 else None)


def analyze(root,records,micro):
    root=Path(root);n=len(records)
    assert n>0
    x=np.asarray([v['features'] for v in records],dtype=np.float32)
    meta=[{k:v[k] for k in ('prompt','split','T','block','position','reached','argmax_mismatch','tv_raw')} for v in records]
    groups=np.asarray([v['prompt'] for v in records]);temperature=np.asarray([v['T'] for v in records]);reached=np.asarray([v['reached'] for v in records],dtype=bool)
    partitions={s:np.asarray([v['split']==s for v in records]) for s in ('train','calibration','holdout')}
    assert not(set(groups[partitions['train']]) & set(groups[partitions['holdout']]))
    assert not(set(groups[partitions['calibration']]) & set(groups[partitions['holdout']]))
    output=[];parameters={};policies=[];curves=[]
    rand=np.random.default_rng(20261009)
    for t in (0,1):
        raw_mismatch=np.asarray([v['argmax_mismatch'] if t==0 else v['tv_raw']>=.10 for v in records],dtype=bool)
        useful=np.asarray([v['greedy_fixed'] if t==0 else v['overlap_gain']>.005 for v in records],dtype=bool)
        gain=np.asarray([float(v['greedy_fixed']) if t==0 else max(v['overlap_gain'],0.) for v in records])
        for task,y in (('MISMATCH',raw_mismatch),('BENEFICIAL_REPAIR',useful)):
            train=partitions['train'] & (temperature==t) & reached
            calibration=partitions['calibration'] & (temperature==t)
            holdout=partitions['holdout'] & (temperature==t)
            if train.sum()<32 or len(np.unique(y[train]))<2:
                output.append(dict(T=t,task=task,status='insufficient_training_classes',states=int(train.sum())))
                continue
            candidates={'CONFIDENCE_1_MINUS_MAXQ':1.-x[:,0],'ENTROPY':x[:,3],'ONE_MINUS_TOP1_TOP2_MARGIN':1.-x[:,2],
                'POSITION_ONLY_CONTROL':np.asarray([v['position']/14. for v in records]),
                'CONSTANT':np.full(n,float(y[train].mean()))}
            for name,cols,kind in (('LOGISTIC_Q_STATS',45,'linear'),('LOGISTIC_Q_STATS_MOMENT',109,'linear'),('MLP_Q_STATS_MOMENT',109,'mlp')):
                candidates[name],parameter=fit(x[:,:cols],y,train,kind)
                parameters[f'T{t}_{task}_{name}']=dict(columns=cols,**parameter)
            for name,score in candidates.items():
                cal_m=metrics(y[calibration&reached],score[calibration&reached])
                reached_m=metrics(y[holdout&reached],score[holdout&reached])
                all_m=metrics(y[holdout],score[holdout])
                per_position={str(pos):metrics(y[holdout & (np.asarray([v['position'] for v in meta])==pos)],score[holdout & (np.asarray([v['position'] for v in meta])==pos)]) for pos in range(15)}
                # Prompt-cluster resampling keeps all blocks/positions together.
                keep=holdout&reached;held_groups=np.unique(groups[keep]);auc_boot=[]
                for _ in range(100):
                    selected=rand.choice(held_groups,len(held_groups),replace=True)
                    unique,counts=np.unique(selected,return_counts=True);lookup=dict(zip(unique,counts))
                    weights=np.asarray([lookup.get(g,0) for g in groups[keep]])
                    auc=metrics(y[keep],score[keep],weights)['AUROC']
                    if auc is not None:auc_boot.append(auc)
                interval=np.quantile(auc_boot,[.025,.975]).tolist() if auc_boot else None
                entry=dict(T=t,task=task,model=name,calibration_reached=cal_m,holdout_reached=reached_m,holdout_all=all_m,
                    prompt_cluster_AUROC_95_interval=interval,per_position=per_position)
                if name.startswith(('LOGISTIC','MLP')):
                    entry['holdout_reached_Brier']=float(np.mean((score[keep]-y[keep])**2))
                output.append(entry)
                if name in ('CONSTANT','POSITION_ONLY_CONTROL'):continue
                # Calibrate only. Maximum skip subject to <=1% helpful-state omissions
                # and <=5% missed positive same-prefix utility, on reached states.
                scan=np.unique(np.r_[-1e-9,np.quantile(score[calibration],np.linspace(0,1,101)),1.000001])
                rows=[coverage(score,z,calibration,meta,gain) for z in scan]
                allowed=[r for r in rows if r['beneficial_state_false_skip']<=.01 and r['reached_positive_utility_proxy_lost_fraction']<=.05]
                chosen=max(allowed,key=lambda r:r['state_skip_fraction'])
                test=coverage(score,chosen['threshold'],holdout,meta,gain)
                policies.append(dict(T=t,task=task,model=name,calibration=chosen,holdout=test,
                    rule='repair when score>=threshold; threshold picked only on calibration',effect_scope='fixed full-head prefixes and observed-reach proxy; no native AL/TPS inference'))
                curves.append(dict(T=t,task=task,model=name,calibration_risk_coverage=rows))
    labels=dict(states=n,prompts=len(set(groups)),reached_states=int(reached.sum()),
        splits={s:dict(prompts=len(set(groups[m])),states=int(m.sum()),reached_states=int((m&reached).sum())) for s,m in partitions.items()},
        tv_sensitivity={str(t):{str(c):float(np.mean([v['tv_raw']>=c for v in records if v['T']==t and v['reached']])) for c in (.05,.10,.20)} for t in (0,1)},
        greedy_fixed=sum(v['greedy_fixed'] for v in records if v['T']==0 and v['reached']),greedy_harmed=sum(v['greedy_harmed'] for v in records if v['T']==0 and v['reached']))
    result=dict(status='complete',diagnostic_only=True,formal_speed_evidence=False,labels=labels,classifiers=output,policies=policies,micro=micro,
        metrics_definition='AUROC groups score ties exactly; AP is non-interpolated average precision; no trapezoidal PR-area label',
        limitations=['GSM8K panel previously used for development; prompt-disjoint research holdout is not a sealed final test.',
            'Target p follows actual full-head proposals. Skipping repair can change subsequent prefixes; fixed-state utilities are not native AL gains.',
            'Finite q feature families and small classifiers cannot prove information sufficiency or impossibility.',
            'Shared GPU operator timings are diagnostic, not clean service throughput.',
            'Coverage at C1 cannot be exponentiated into C40 speed; compaction/routing/Graph costs need native confirmation.',
            'This experiment never bypasses target verification or changes the repair training.'],hidden_or_logit_cache_files=0,utc=now())
    dump(root/'RESULTS.json',result);dump(root/'risk_coverage.json',curves);dump(root/'classifier_parameters.json',parameters)
    lines=['当前q-only mismatch / 修复价值预测预实验','当前联合长训权重；三层融合头；block16；实际native T0/T1。','仅q作为预测输入；按提示80/24/24划分，状态/两温度不跨分组；只用校准组选阈值。',
        f"实际采集：{labels['prompts']}提示，{n}状态，{labels['reached_states']}实际到达状态。",
        'T | 任务 | 模型 | 留出到达AUROC | AP | 阳性率 | 提示bootstrap AUROC区间']
    for r in output:
        if 'holdout_reached' not in r:continue
        m=r['holdout_reached'];interval=r['prompt_cluster_AUROC_95_interval']
        fmt=lambda v:'不足两类' if v is None else f'{v:.4f}'
        lines.append(f"{r['T']} | {r['task']} | {r['model']} | {fmt(m['AUROC'])} | {fmt(m['AP'])} | {m['positive_rate']:.4f} | {interval}")
    lines+=['','T | 任务 | 模型 | 留出状态跳过 | 整块跳过 | 漏掉正收益代理 | 到达位置正收益代理损失']
    for r in policies:
        v=r['holdout'];loss=v['positive_utility_proxy_lost_fraction']
        lines.append(f"{r['T']} | {r['task']} | {r['model']} | {v['state_skip_fraction']:.2%} | {v['whole_block_skip_fraction']:.2%} | {loss} | {v['reached_positive_utility_proxy_lost_fraction']:.2%}")
    lines+=['','这些是固定当前真实前缀上的诊断与收益代理，不是跳过修复后的native AL/TPS。','判断成本（独立CUDA Graph微测；共享卡污染不作正式测速）：'+json.dumps(micro,ensure_ascii=False),
        '参考文献及机制边界见research/q_only_skip_probe_20261009_v1/SOURCE_CHECKS.txt。','特征、hidden、logits和概率仅在进程内存；只保存统计、曲线和小分类器权重。','下一步只有在校准/留出显示可用skip比例和低漏修后，才考虑独立闭环gate测试；torch.where双支计算不能当作跳过实现。']
    (root/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')
