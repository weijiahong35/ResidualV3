"""Publish isolated E2 paired results and refresh the locked experiment ledger."""
from pathlib import Path
import fcntl, hashlib, json, sys

CODE=Path(__file__).resolve().parent
BASE=CODE.parents[2]
ROOT=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(CODE/'native_runtime'))
from runtime.utils import dump, now


def read(path):
    return json.loads(path.read_text())


def main():
    plan=read(ROOT/'plan.json')
    status=read(ROOT/'status.json')
    lines=['# E2：DSpark代码与对话六套数据重测','',
        '只测E2独立单训DFlash与E2联合主体＋完整三层FULL15_NOFILL；两臂主体权重不同。',
        'C1/C32，T0/T1，固定15 proposal；BF16、batch-invariant、CUDA Graph，无尾部截断。',
        '最大生成2048；五套数据context4096，Arena-Hard-v2 context12288以容纳原始长输入。',
        '固定真实KV：C1 52GiB、C32 44GiB，smoke与formal相同，utilization=0.92。',
        '至少78000MiB空闲且宿主无GPU进程才启动；最多同时占两张空卡。',
        'AC为包含bonus的平均接受长度；吞吐为整窗输出token/s，不是任务准确率。','',
        f'当前状态：{status["status"]}。新数据与作者的固定提交、抽样数、抽样顺序、首轮处理、nonthinking、2048生成上限、种子980406对齐。',
        '引擎仍为本项目vLLM，proposal仍为15（DSpark原论文为7），并额外测T0及C32；这不是DSpark完整方法复现。',
        '未重复扩充题目；MBPP作者选出的256个来源行中有255个不同题面，按原样保留。','',
        '[即时控制器log](parallel/controller.log) · [状态](status.json) · [数据审计](DATASET_AUDIT.json) · [固定协议](plan.json)','',
        '| 领域 | 数据集 | 每窗来源行数 | 不同题面数 | 上下文上限 |',
        '|---|---|---:|---:|---:|']
    for ds,spec in plan['datasets'].items():
        lines.append(f'| {spec["domain"]} | {ds} | {spec["count"]} | {spec["unique_prompt_texts"]} | {spec["context_limit"]} |')
    lines += ['', '每个模型×温度×并发使用同一组2000个来源行；全批共48正式窗口、16000个请求。',
        '旧HumanEval/MT-Bench使用max2096、seed42，保留为历史；本批按新请求重新测，不将跨批TPS相除。', '',
        '[作者TASKS与默认参数](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py)；',
        '[作者首轮处理与抽样](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py)。','']
    results={};valid_windows=0
    for ds,spec in plan['datasets'].items():
        lines += ['## '+ds,'',
            f'[C1 smoke]({ds}_C1_smoke.log) · [C1 formal]({ds}_C1_formal.log) · [C32 smoke]({ds}_C32_smoke.log) · [C32 formal]({ds}_C32_formal.log)','',
            '| 数据集 | 设置 | AC：单训→联合头 | AC增长 | 吞吐：单训→联合头 | 吞吐增长 |',
            '|---|---|---:|---:|---:|---:|']
        for c in plan['concurrency']:
            path=ROOT/ds/f'C{c}/formal/summary.json'
            windows={}
            if path.exists():
                summary=read(path)
                windows={(w['arm'],w['T']):w for w in summary['windows']
                         if w.get('isolation_observed') and w.get('speed_evidence') and not w.get('performance_contaminated')}
                valid_windows+=len(windows)
            for t in plan['temperatures']:
                if not all((a,t) in windows for a in plan['arms']):
                    lines.append(f'| {ds} | C{c} T{t} | 待完成 | — | 待完成 | — |')
                    continue
                a,b=[windows[(arm,t)] for arm in plan['arms']]
                x,y=a['native_metrics']['acceptance_length'],b['native_metrics']['acceptance_length']
                u,v=a['tokens_per_second'],b['tokens_per_second']
                row=dict(dataset=ds,concurrency=c,T=t,baseline_AC=x,head_AC=y,
                         AC_growth_pct=(y/x-1)*100,baseline_TPS=u,head_TPS=v,TPS_growth_pct=(v/u-1)*100)
                results[f'{ds}_C{c}_T{t}']=row
                lines.append(f'| {ds} | C{c} T{t} | {x:.3f}→{y:.3f} | {row["AC_growth_pct"]:+.2f}% | {u:.1f}→{v:.1f} | {row["TPS_growth_pct"]:+.2f}% |')
        lines.append('')
    (ROOT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')
    dump(ROOT/'COMPARISON.json',results)
    dump(ROOT/'progress.json',dict(status=status['status'],valid_formal_windows=valid_windows,
         paired_comparisons=len(results),formal_windows_expected=48,paired_comparisons_expected=24,utc=now()))
    signature=hashlib.sha256(json.dumps(dict(status=status['status'],results=results),sort_keys=True).encode()).hexdigest()
    saved=ROOT/'ledger_progress.json'
    if saved.exists() and read(saved).get('signature')==signature:return
    maintenance=BASE/'residualV3_main_v1/maintenance/experiment_ledger'
    sys.path.insert(0,str(maintenance))
    import ledger
    key=ROOT.relative_to(BASE).as_posix()
    with (maintenance/'ledger.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        registry=read(maintenance/'registry.json')
        registry['experiments'][key]=dict(mode='nonthinking',title='E2代码与对话：DSpark六套数据C1/C32重测',
            purpose='按用户要求补齐DSpark代码与对话的六套评测；只使用E2独立与联合三层权重。',
            comparison='E2独立单训DFLASH vs E2联合主体＋三层FULL15_NOFILL，无专家路由、无截断。',
            settings='HumanEval164/MBPP256/LiveCodeBench500/MT-Bench80/Alpaca500/Arena-Hard-v2 500；作者seed980406抽样与首轮，max2048；C1/C32 T0/T1。KV固定52/44GiB，util0.92；Arena上下文12288、其余4096；两张空卡上限。',
            conclusion=f'状态{status["status"]}；有效正式窗口{valid_windows}/48，成对对照{len(results)}/24。原结果保留，新结果只按本批同窗比较。',
            formal_sources=[f'{ds}/C{c}/formal/summary.json' for ds in plan['datasets'] for c in plan['concurrency']],
            evidence=[f'{key}/{name}' for name in ['RESULTS_CN.md','plan.json','DATASET_AUDIT.json','READY_AUDIT.json','status.json','parallel/controller.log']])
        ledger.atomic_write(maintenance/'registry.json',json.dumps(registry,ensure_ascii=False,indent=2)+'\n')
        ledger.refresh()
    dump(saved,dict(signature=signature,utc=now()))


if __name__=='__main__':main()
