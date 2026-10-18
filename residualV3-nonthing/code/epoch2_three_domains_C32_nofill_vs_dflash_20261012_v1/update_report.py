"""Read-only C32 report writer; isolated paired windows only."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve()
plan=json.loads((root/'plan.json').read_text())
lines=['# E2 三领域 C32 正式补测','', '独立单训 DFlash vs 联合训练 DFlash＋FULL15_NOFILL；固定15 proposal，无尾部截断。', 'BF16、batch-invariant、CUDA Graph；T0/T1，max2096/context4096。新冻结代码，权重与既有正式版一致。AC为平均接受长度，不是准确率；吞吐单位token/s。','', '数量：GSM8K=1319、HumanEval=164、MT-Bench=80（首轮问题），与原C40题目及顺序相同，无重复扩充。C32 formal KV44GiB、smoke38GiB。所有GPU0–7允许，只用宿主无进程且free≥78000MiB的空卡。','', '[控制器log](parallel/controller.log)、[当前状态](status.json)、[代码与设置](plan.json)。','']
results={}
for ds,spec in plan['datasets'].items():
    lines += ['## '+ds,'',f'[smoke log]({ds}_C32_smoke.log) · [formal log]({ds}_C32_formal.log)','', '| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |','|---|---|---:|---:|---:|---:|']
    path=root/ds/'C32/formal/summary.json'
    if not path.exists():
        lines+=['等待匹配smoke与正式窗完成。',''];continue
    s=json.loads(path.read_text());windows={(w['arm'],w['T']):w for w in s['windows'] if w['isolation_observed'] and not w['performance_contaminated']}
    for t in [0,1]:
        if not all((a,t) in windows for a in plan['arms']):continue
        a,b=[windows[(arm,t)] for arm in plan['arms']]
        x,y=a['native_metrics']['acceptance_length'],b['native_metrics']['acceptance_length'];u,v=a['tokens_per_second'],b['tokens_per_second']
        row=dict(dataset=ds,concurrency=32,T=t,baseline_AC=x,head_AC=y,AC_growth_pct=(y/x-1)*100,baseline_TPS=u,head_TPS=v,TPS_growth_pct=(v/u-1)*100)
        results[f'{ds}_C32_T{t}']=row
        lines.append(f'| {ds} | C32 T{t} | {x:.3f}→{y:.3f} | {row["AC_growth_pct"]:+.2f}% | {u:.1f}→{v:.1f} | {row["TPS_growth_pct"]:+.2f}% |')
    lines+=['']
(root/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
(root/'COMPARISON.json').write_text(json.dumps(results,indent=2)+'\n')
