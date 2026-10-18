"""Paired pre/post-q diagnostic on common native labels, never rollout TPS."""
from pathlib import Path
import json
import numpy as np
from runtime.utils import dump, now
from .q_probe_legacy_metrics import fit, metrics


def block_groups(records, subset):
    groups = {}
    for i in np.flatnonzero(subset):
        v = records[i]
        groups.setdefault((v['prompt'], v['T'], v['block']), []).append(int(i))
    out = []
    for ids in groups.values():
        ids.sort(key=lambda i: records[i]['position'])
        assert [records[i]['position'] for i in ids] == list(range(15))
        assert len({records[i]['accepted_proposals'] for i in ids}) == 1
        out.append(ids)
    return out


def cutoff(score, stage, threshold, patience=1, grace=0, minimum=8):
    """PRE j is seen before repairing j; POST j after repairing j (j zero-based)."""
    consecutive = 0
    pending = None
    for j in range(15):
        done = j if stage == 'PRE' else j + 1
        if done < minimum:
            continue
        if score[j] >= threshold:
            consecutive += 1
            if pending is None and consecutive >= patience:
                pending = done + grace
            if pending is not None and done >= pending:
                return min(done, 15)
        else:
            # Conservative: recovery cancels the candidate stopping window.
            consecutive = 0
            pending = None
    return 15


def opportunity(score, subset, records, stage, threshold, patience, grace):
    groups = block_groups(records, subset)
    cuts = [cutoff(score[ids], stage, threshold, patience, grace) for ids in groups]
    accepted = [records[ids[0]]['accepted_proposals'] for ids in groups]
    lost = sum(max(k-c, 0) for k,c in zip(accepted, cuts))
    return dict(threshold=float(threshold), blocks=len(groups),
                mean_repair_positions=float(np.mean(cuts)),
                repair_positions_saved_fraction=float(np.mean([(15-c)/15 for c in cuts])),
                removed_full_path_accepted_proposals=lost,
                full_path_accepted_opportunity_lost_fraction=lost/max(sum(accepted),1),
                cutoff_counts={str(c):cuts.count(c) for c in range(16)},
                scope='Full-repair trajectory opportunity proxy; raw suffix remains proposed. Not counterfactual AL loss.')


def paired_bootstrap(y, pre, post, prompt, subset, seed=20261009):
    groups = np.unique(prompt[subset])
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(200):
        picked = rng.choice(groups, len(groups), replace=True)
        ids, counts = np.unique(picked, return_counts=True)
        lookup = dict(zip(ids, counts))
        weights = np.asarray([lookup.get(p,0) for p in prompt[subset]])
        a = metrics(y[subset], pre[subset], weights)['AUROC']
        b = metrics(y[subset], post[subset], weights)['AUROC']
        if a is not None and b is not None:
            values.append(b-a)
    return np.quantile(values,[.025,.975]).tolist() if values else None


def analyze(root, records, micro):
    root = Path(root)
    pre = np.asarray([r['features_pre'] for r in records],dtype=np.float32)
    post = np.asarray([r['features_post'] for r in records],dtype=np.float32)
    assert pre.shape == post.shape and pre.shape[1] == 45
    n = len(records)
    prompt = np.asarray([r['prompt'] for r in records])
    temp = np.asarray([r['T'] for r in records])
    position = np.asarray([r['position'] for r in records])
    reached = np.asarray([r['reached'] for r in records],dtype=bool)
    partitions = {s:np.asarray([r['split']==s for r in records]) for s in ('train','calibration','holdout')}
    sets = {s:set(prompt[m]) for s,m in partitions.items()}
    assert not (sets['train'] & sets['calibration'] or sets['train'] & sets['holdout'] or sets['calibration'] & sets['holdout'])
    prefix_scores = {}
    for label,x in (('PRE',pre),('POST',post)):
        z = np.zeros(n)
        for ids in block_groups(records,np.ones(n,dtype=bool)):
            z[ids] = 1-np.exp(np.cumsum(np.log(x[ids,0].clip(1e-30,1))))
        prefix_scores[label] = z
    output = []; paired = []; policies = []; parameters = {}
    for t in (0,1):
        common = temp==t
        labels = {
            'RAW_MISMATCH':np.asarray([r['argmax_mismatch'] if t==0 else r['tv_raw']>=.10 for r in records]),
            'REPAIRED_MISMATCH':np.asarray([r['repaired_argmax_mismatch'] if t==0 else r['tv_post']>=.10 for r in records]),
            'NATIVE_REJECTION':np.asarray([r['native_reject'] for r in records]),
            'NO_ACCEPTED_SUFFIX_AFTER_CURRENT':np.asarray([r['accepted_suffix_after_current']==0 for r in records])
        }
        for task,y in labels.items():
            local = task != 'NO_ACCEPTED_SUFFIX_AFTER_CURRENT'
            train = partitions['train'] & common & (reached if local else position>=5)
            calibration = partitions['calibration'] & common
            holdout = partitions['holdout'] & common
            if len(np.unique(y[train]))<2 or train.sum()<32:
                output.append(dict(T=t,task=task,status='insufficient_classes'));continue
            scores = {'POSITION_ONLY':position/14., 'CONSTANT':np.full(n,float(y[train].mean()))}
            for label,x in (('PRE',pre),('POST',post)):
                scores[label+'_CONFIDENCE'] = 1-x[:,0]
                scores[label+'_MARGIN'] = 1-x[:,2]
                scores[label+'_ENTROPY'] = x[:,3]
                scores[label+'_PREFIX_MAXQ_PRODUCT'] = prefix_scores[label]
                score,param = fit(x,y,train,'linear')
                scores[label+'_LOGISTIC'] = score
                parameters[f'T{t}_{task}_{label}_LOGISTIC'] = param
            score,param = fit(np.concatenate([pre,post,post-pre],1),y,train,'linear')
            scores['JOINT_CHANGE_LOGISTIC'] = score
            parameters[f'T{t}_{task}_JOINT_CHANGE_LOGISTIC'] = param
            for name,score in scores.items():
                masks = dict(reached=holdout&reached,all=holdout,tail_all=holdout&(position>=7),tail_reached=holdout&reached&(position>=7))
                row = dict(T=t,task=task,signal=name,feature_timing='pre-repair' if name.startswith('PRE') else 'post-repair' if name.startswith(('POST','JOINT')) else 'control',
                           evaluations={key:metrics(y[m],score[m]) for key,m in masks.items()},
                           per_position={str(j+1):metrics(y[holdout&(position==j)],score[holdout&(position==j)]) for j in range(15)})
                if name.endswith('LOGISTIC'):
                    row['holdout_reached_Brier'] = float(np.mean((score[holdout&reached]-y[holdout&reached])**2))
                output.append(row)
            for kind in ('CONFIDENCE','LOGISTIC','PREFIX_MAXQ_PRODUCT'):
                a=scores['PRE_'+kind];b=scores['POST_'+kind]
                for zone,subset in (('reached',holdout&reached),('tail_all',holdout&(position>=7))):
                    am=metrics(y[subset],a[subset]);bm=metrics(y[subset],b[subset])
                    paired.append(dict(T=t,task=task,signal_family=kind,zone=zone,pre=am,post=bm,
                                       post_minus_pre_AUROC=None if am['AUROC'] is None or bm['AUROC'] is None else bm['AUROC']-am['AUROC'],
                                       prompt_cluster_paired_AUROC_difference_95_interval=paired_bootstrap(y,a,b,prompt,subset)))
            if task!='NO_ACCEPTED_SUFFIX_AFTER_CURRENT':continue
            for stage in ('PRE','POST'):
                for kind in ('CONFIDENCE','LOGISTIC','PREFIX_MAXQ_PRODUCT'):
                    score=scores[stage+'_'+kind]
                    scan=np.unique(np.r_[-1e-9,np.quantile(score[calibration],np.linspace(0,1,41)),1.000001])
                    for patience,grace in ((1,0),(2,0),(2,2),(2,4)):
                        rows=[opportunity(score,calibration,records,stage,z,patience,grace) for z in scan]
                        allowed=[r for r in rows if r['full_path_accepted_opportunity_lost_fraction']<=.05]
                        chosen=max(allowed,key=lambda r:r['repair_positions_saved_fraction'])
                        policies.append(dict(T=t,signal=stage+'_'+kind,stage=stage,patience=patience,grace=grace,minimum_repair=8,
                            calibration=chosen,holdout=opportunity(score,holdout,records,stage,chosen['threshold'],patience,grace),
                            selection_rule='Calibrate maximum skipped positions subject to <=5% full-path accepted opportunity removed; proxy only, not repair-gain preservation.'))
    counts=dict(states=n,prompts=len(set(prompt)),blocks=n//15,reached_states=int(reached.sum()),
                splits={s:dict(prompts=len(sets[s]),states=int(m.sum())) for s,m in partitions.items()})
    result=dict(status='complete',diagnostic_only=True,formal_speed_evidence=False,labels=counts,classifiers=output,
                paired_comparisons=paired,policies=policies,micro=micro,
                common_label_definition=dict(T0='argmax(repaired q) != argmax(target p)',T1='TV(target p,repaired q)>=0.10',
                    NATIVE_REJECTION='First actual rejected proposal, evaluated only on actual reached positions',
                    NO_ACCEPTED_SUFFIX_AFTER_CURRENT='Actual full-repair accepted K <= number of proposals through current position'),
                limitations=['Same native state and target prefix for PRE/POST. No deployed cutoff in this probe.',
                    'GSM8K and this prompt-disjoint research holdout have previous development use; not a sealed final test.',
                    'Full-path suffix opportunity removed is not STOP-vs-FULL causal loss. Raw suffix still proposes tokens.',
                    'Prefix max-q product is a heuristic, not a product of calibrated conditional acceptance probabilities.',
                    'Native T1 rejection is stochastic; probability-thresholded mismatch and realized rejection are different labels.',
                    'Signal micro timings exclude repair, CPU/GPU routing and batching; no native TPS claim.'],hidden_or_logit_cache_files=0,utc=now())
    result['paired_summary'] = [dict(T=r['T'],task=r['task'],signal=r['signal_family'],zone=r['zone'],
        pre_AUROC=r['pre']['AUROC'],post_AUROC=r['post']['AUROC'],delta_AUROC=r['post_minus_pre_AUROC'],
        paired_prompt_interval=r['prompt_cluster_paired_AUROC_difference_95_interval'])
        for r in paired if r['task'] in ('REPAIRED_MISMATCH','NO_ACCEPTED_SUFFIX_AFTER_CURRENT')]
    dump(root/'RESULTS.json',result);dump(root/'classifier_parameters.json',parameters)
    lines=['修复前q与修复后q：同状态配对诊断','冻结当前联合长训主体、三层头及basis；non-thinking；block16=15proposal。',
        f"{counts['prompts']}提示、{counts['blocks']}块、{n}状态、{counts['reached_states']}实际到达状态。",
        '共同标签：T0修复后argmax错误，T1修复后TV>=0.10；另列native实际拒绝及后续接受机会。',
        'T | 任务 | 特征 | 留出到达AUROC | AP | 第8-15位置AUROC（包含未到达）']
    def fmt(v):return '不足两类' if v is None else f'{v:.4f}'
    for row in output:
        if 'evaluations' not in row:continue
        a=row['evaluations']['reached'];b=row['evaluations']['tail_all']
        lines.append(f"{row['T']} | {row['task']} | {row['signal']} | {fmt(a['AUROC'])} | {fmt(a['AP'])} | {fmt(b['AUROC'])}")
    lines+=['','修复后减修复前AUROC差值，提示级配对bootstrap区间（研究留出）：']
    for r in paired:
        if r['task']=='RAW_MISMATCH':continue
        lines.append(f"T{r['T']} {r['task']} {r['signal_family']} {r['zone']}: {fmt(r['post_minus_pre_AUROC'])}, {r['prompt_cluster_paired_AUROC_difference_95_interval']}")
    lines+=['','以下是完整修复路径上的尾部机会代理，绝不是截断后的native AL/TPS：',
            'T | 信号 | patience | 再修位置 | 平均修位置 | 省修比例 | 删去的完整路径接受机会比例']
    for r in policies:
        z=r['holdout']
        lines.append(f"{r['T']} | {r['signal']} | {r['patience']} | {r['grace']} | {z['mean_repair_positions']:.3f} | {z['repair_positions_saved_fraction']:.2%} | {z['full_path_accepted_opportunity_lost_fraction']:.2%}")
    lines+=['','判断算子Graph微测：'+json.dumps(micro,ensure_ascii=False),
        'POST必须支付当前位置repair成本，最多省后续位置；PRE有机会连当前位置一起省。',
        '特征/概率/hidden/logit/逐状态标签不落盘；只保存聚合指标、曲线及小分类器参数。',
        '部署阈值与正式AL/TPS尚未测。两种信号均需后续STOP与FULL分支标签和真实闭环验证。']
    (root/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')
