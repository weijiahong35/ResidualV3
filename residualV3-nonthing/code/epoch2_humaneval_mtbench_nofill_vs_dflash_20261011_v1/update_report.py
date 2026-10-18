"""Read new cross-domain evidence and maintain the locked experiment registry."""
from pathlib import Path
import datetime, fcntl, json, os, subprocess, sys

ROOT = Path(sys.argv[1]).resolve()
BASE = ROOT.parents[2]
LEDGER = BASE / 'residualV3_main_v1/maintenance/experiment_ledger'


def read(p): return json.loads(p.read_text())


def write(p, value):
    tmp = p.with_name(p.name + f'.tmp.report.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def main():
    state = read(ROOT / 'status.json')
    plan = read(ROOT / 'plan.json')
    windows, pairs = [], []
    for dataset in plan['datasets']:
        for c in plan['concurrency']:
            f = ROOT / dataset / f'C{c}/formal/summary.json'
            if not f.exists(): continue
            data = read(f)
            by = {(w['arm'], w['T']): w for w in data['windows']}
            windows.extend(dict(w, dataset=dataset, C=c) for w in data['windows'])
            for t in plan['temperatures']:
                if not all((a, t) in by for a in plan['arms']): continue
                d, h = [by[a, t] for a in plan['arms']]
                da, ha = [r['native_metrics']['acceptance_length'] for r in (d, h)]
                dt, ht = [r['tokens_per_second'] for r in (d, h)]
                clean = all(r.get('isolation_observed') and r.get('speed_evidence')
                            and not r.get('performance_contaminated') for r in (d, h))
                pairs.append(dict(dataset=dataset, C=c, T=t, requests=d['requests'],
                    DFLASH_AL=da, FULL15_NOFILL_AL=ha, AL_percent=100 * (ha / da - 1),
                    DFLASH_TPS=dt, FULL15_NOFILL_TPS=ht, TPS_percent=100 * (ht / dt - 1),
                    isolated=clean, valid_speed_evidence=clean, source=str(f.relative_to(ROOT))))
    write(ROOT / 'COMPARISON.json', dict(status=state['status'], windows_completed=len(windows),
         windows_expected=16, rows=pairs, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    desc = f"2026-10-11 E2跨域对照：HumanEval164题、MT-Bench80题，均为DSpark第一轮全量原题。状态{state['status']}，正式窗{len(windows)}/16。"
    if pairs:
        desc += '；'.join(f"{x['dataset']} C{x['C']} T{x['T']} AL{x['AL_percent']:+.2f}% TPS{x['TPS_percent']:+.2f}% 隔离{x['isolated']}" for x in pairs) + '。'
    text = '# E2 HumanEval / MT-Bench：独立DFlash对联合完整头\n\n' + desc + '\n\n'
    text += ('只测DFLASH（第二轮独立单训主体）和FULL15_NOFILL（第二轮联合主体＋三层修复头）。不同主体权重的整体模型对比；固定15个proposal，不启用尾部截断或辅助预测器。推理runtime与已完成GSM8K正式版逐文件相同，仅新建评测入口、数据协议及控制器。\n\n'
             'HumanEval：C1/C40各164道唯一题。MT-Bench：C1/C40各80道唯一题，按DSpark的AL评测代码只取第一轮；不是完整两轮MT-Bench质量评分。原题不循环补足200/1319。每个数据集均T0/T1、两臂、单轮；共16个正式窗/1952次正式请求。C40有有限面板的启动和尾部排空，不能解释为1319请求持续负载。\n\n'
             '保留此前max_new=2096/context4096、seed42+请求序号、BF16、batch-invariant、CUDA Graph、block16=anchor+15 proposal。DSpark默认T1/max2048/seed980406，本次对齐题量、官方prompt、第一轮与nonthinking口径，保留项目原推理设置；没有复现其权重、训练或离线推理引擎。\n\n'
             '先在实际数据prompt上做smoke合同核验，再正式计时；两臂同卡同窗设置。C1 KV52GiB；C40 smoke38GiB、正式44GiB。只用宿主无进程且空闲至少78000MiB的0/3/4/5/6/7号卡；GPU1/2排除。AL是接受长度，包含bonus，不是答题准确率；TPS是实际生成token数除以HTTP整窗墙钟时间。\n\n'
             '| 数据集 | 设置 | 单训 AL | 联合头 AL | AL 增长 | 单训 TPS | 联合头 TPS | TPS 增长 | 隔离 |\n'
             '|---|---|---:|---:|---:|---:|---:|---:|---|\n')
    for r in pairs:
        text += f"| {r['dataset']} | C{r['C']} T{r['T']} | {r['DFLASH_AL']:.6f} | {r['FULL15_NOFILL_AL']:.6f} | {r['AL_percent']:+.2f}% | {r['DFLASH_TPS']:.3f} | {r['FULL15_NOFILL_TPS']:.3f} | {r['TPS_percent']:+.2f}% | {r['isolated']} |\n"
    if not pairs: text += '\n尚无完整正式对照，不推算跨域收益。\n'
    text += '\n当前状态：' + json.dumps({k: v for k, v in state.items() if k not in ('host_checks', 'command')}, ensure_ascii=False) + '\n\n'
    text += '[状态](status.json)、[宿主GPU检查](HOST_GPU_GATE.json)、[准备审计](PREFLIGHT.json)、[DSpark口径](DSPARK_PROTOCOL_AUDIT.json)、[原始比较](COMPARISON.json)、[控制器即时log](controller.log)。\n\n'
    for dataset in plan['datasets']:
        for c in plan['concurrency']:
            text += f"{dataset} C{c}：[smoke log]({dataset}_C{c}_smoke.log)、[正式log]({dataset}_C{c}_formal.log)。\n\n"
    if (ROOT / 'FINAL_AUDIT.json').exists(): text += '[最终审计](FINAL_AUDIT.json)。\n'
    (ROOT / 'RESULTS_CN.md').write_text(text)
    with (LEDGER / 'ledger.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reg = read(LEDGER / 'registry.json')
        reg['experiments'][str(ROOT.relative_to(BASE))] = dict(mode='nonthinking',
            title='E2 HumanEval/MT-Bench：独立DFlash与联合完整修复头',
            purpose='验证代码与聊天领域的native接受长度及端到端吞吐收益。',
            comparison='E2独立单训DFLASH vs E2联合主体+完整FULL15_NOFILL头；仅两臂，无尾部截断/辅助预测。',
            settings='DSpark全量HumanEval164/MT-Bench80第一轮唯一题；C1/C40、T0/T1、max2096/context4096、BF16、batch-invariant/CUDA Graph；单轮；先实际prompt smoke；仅空卡，排除1/2。',
            conclusion=desc, formal_sources=[f'{d}/C{c}/formal/summary.json' for d in plan['datasets'] for c in plan['concurrency']],
            evidence=[str((ROOT / f).relative_to(BASE)) for f in ('RESULTS_CN.md','COMPARISON.json','plan.json','PREFLIGHT.json','DSPARK_PROTOCOL_AUDIT.json','status.json')])
        prefix = '2026-10-11 E2跨域对照：'
        for mode in ('nonthinking', 'thinking'):
            notes = reg['notes'][mode]
            notes[:] = [x for x in notes if not x.startswith(prefix)]
            notes.append(desc + (' 本轮仅Nonthinking；Thinking没有新增训练/评测。' if mode == 'thinking' else ''))
        write(LEDGER / 'registry.json', reg)
    subprocess.run([sys.executable, '-B', str(LEDGER / 'ledger.py'), 'refresh'],
                   check=True, stdout=subprocess.DEVNULL)
    print(desc, flush=True)


if __name__ == '__main__': main()
