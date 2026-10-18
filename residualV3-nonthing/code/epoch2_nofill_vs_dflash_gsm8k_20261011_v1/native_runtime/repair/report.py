"""C1 per-ablation paired RAW comparison, without treating repeated rounds as samples."""
import argparse,json
from pathlib import Path
from runtime.utils import dump,sha,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);out=ap.parse_args().output
    p=json.loads((out/'preregistration.json').read_text());summary=json.loads((out/'native/C1_single/summary.json').read_text());rows=summary['windows']
    assert summary['status']=='complete' and len(rows)==4
    assert all(w['speed_evidence'] and w['isolation_observed'] and not w['smoke'] for w in rows)
    init=json.loads((out/'training/EMBEDDING/initialization.json').read_text())
    paired=[];lines=[f"# {p['variant']} — EMBEDDING C1",'',
        '所有 repair 参数重新随机初始化；RAW 与修复模型同卡、同20条prompt、同seed，T0/T1，4096/8192，mem0.93，CUDA Graph。单轮RAW→EMBEDDING。',
        f"参数量：{init['parameter_count']}；结构：{json.dumps(p['architecture'],ensure_ascii=False)}",'',
        '| T | 轮 | 模型 | 接受长度 | token/s | 相对RAW接受长度 | 相对RAW吞吐 |',
        '|---|---|---|---:|---:|---:|---:|']
    for t in (0,1):
        for rnd in (1,):
            cells={w['arm']:w for w in rows if w['T']==t and w['round']==rnd};assert set(cells)=={'RAW','EMBEDDING'}
            raw=cells['RAW']
            for arm in ('RAW','EMBEDDING'):
                w=cells[arm];ar=w['native_metrics']['acceptance_length']/raw['native_metrics']['acceptance_length'];sr=w['tokens_per_second']/raw['tokens_per_second']
                lines.append(f"| {t} | {rnd} | {arm} | {w['native_metrics']['acceptance_length']:.6f} | {w['tokens_per_second']:.2f} | {(ar-1)*100:+.2f}% | {(sr-1)*100:+.2f}% |")
                if arm!='RAW':paired.append(dict(T=t,round=rnd,arm=arm,raw_acceptance=raw['native_metrics']['acceptance_length'],repair_acceptance=w['native_metrics']['acceptance_length'],raw_tps=raw['tokens_per_second'],repair_tps=w['tokens_per_second'],acceptance_ratio=ar,throughput_ratio=sr))
    lines+=['','接受长度与吞吐分别报告；只测一轮，不报告重复样本显著性。测试panel已有使用历史。训练为既有答案teacher forcing，native为实际采样反馈。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(status='complete',variant=p['variant'],parameter_count=init['parameter_count'],utc=now(),summary_sha256=sha(out/'native/C1_single/summary.json'),paired=paired))
    print('\n'.join(lines))
if __name__=='__main__':main()
