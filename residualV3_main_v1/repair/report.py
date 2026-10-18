"""Same-card C1 comparison; input-table effect separated from RAW speed comparison."""
import argparse,json
from pathlib import Path
from .model import ARMS
from runtime.utils import dump,sha,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();out=a.output
    summary=json.loads((out/'native/C1/summary.json').read_text());rows=summary['windows']
    assert summary['status']=='complete' and len(rows)==12
    assert all(w['speed_evidence'] and w['isolation_observed'] and not w['smoke'] for w in rows)
    inits=[json.loads((out/'training'/arm/'initialization.json').read_text()) for arm in ARMS]
    assert inits[0]['tensor_sha256']==inits[1]['tensor_sha256']
    paired=[]
    lines=['# 硬/软lag-1 + 标量dose：LM-head / embedding 对照','',
      'GPU5，同20 prompts、seed42+index、T0/T1，C1，max4096/total8192，mem.93，CUDA Graph。两轮正反顺序。',
      '两版全部可训练参数从同一随机seed独立初始化，同一在线状态、锚点与答案，独立optimizer；只替换三条token特征的冻结输入表。',
      'direction输出始终使用原LM-head与同一train128-only基底。两个输入表也共用该基底，因此不是embedding投影方式的最优性比较。','',
      '| T | 轮 | 模型 | 接受长度 | token/s | 相对RAW接受长度 | 相对RAW吞吐 |',
      '|---|---|---|---:|---:|---:|---:|']
    for t in (0,1):
        for rnd in (1,2):
            cells={w['arm']:w for w in rows if w['T']==t and w['round']==rnd};assert set(cells)=={'RAW',*ARMS}
            raw=cells['RAW']
            for arm in ('RAW',*ARMS):
                w=cells[arm];ar=w['native_metrics']['acceptance_length']/raw['native_metrics']['acceptance_length'];sr=w['tokens_per_second']/raw['tokens_per_second']
                lines.append(f"| {t} | {rnd} | {arm} | {w['native_metrics']['acceptance_length']:.6f} | {w['tokens_per_second']:.2f} | {(ar-1)*100:+.2f}% | {(sr-1)*100:+.2f}% |")
                if arm!='RAW':paired.append(dict(T=t,round=rnd,arm=arm,raw_acceptance=raw['native_metrics']['acceptance_length'],repair_acceptance=w['native_metrics']['acceptance_length'],raw_tps=raw['tokens_per_second'],repair_tps=w['tokens_per_second'],acceptance_ratio=ar,throughput_ratio=sr))
            lm,emb=cells['LMHEAD'],cells['EMBEDDING']
            paired.append(dict(T=t,round=rnd,arm='EMBEDDING/LMHEAD',acceptance_ratio=emb['native_metrics']['acceptance_length']/lm['native_metrics']['acceptance_length'],throughput_ratio=emb['tokens_per_second']/lm['tokens_per_second']))
    lines+=['','每版4,419,585个可训练参数，dose257个。soft特征固定T=1；verifier仍使用真实采样分布。',
      '只用OUTPUT目标，128条×3epoch、最多1536锚点。训练硬前驱为teacher forcing，native硬前驱为实际采样反馈。',
      '这两个实验同时加入软反馈、简化dose，因此与旧三输入结果不能作单因素因果归因；两版之间只换输入表。',
      '接受长度和吞吐分开判断。测试panel已有开发使用历史；两轮重复不是独立样本，微小差异不等于统计显著。']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(out/'results.json',dict(status='complete',utc=now(),summary_sha256=sha(out/'native/C1/summary.json'),paired=paired));print('\n'.join(lines))
if __name__=='__main__':main()
