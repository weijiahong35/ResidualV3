"""Formal same-card RAW versus no-soft3 summary, retaining contamination flags."""
import argparse
import csv
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output.resolve()
    rows=[]
    for concurrency in [1,40]:
        summary=json.loads((out/f'C{concurrency}/formal/summary.json').read_text())
        assert summary['status']=='complete' and len(summary['windows'])==4
        mapping={(window['arm'],window['T']):window for window in summary['windows']}
        for temperature in [0,1]:
            raw=mapping['RAW',temperature]
            candidate=mapping['N_no_soft_3layer_nonthinking',temperature]
            for window in [raw,candidate]:
                rows.append(dict(C=concurrency,T=temperature,arm=window['arm'],AL=window['native_metrics']['acceptance_length'],TPS=window['tokens_per_second'],
                    AL_vs_RAW_pct=100*(window['native_metrics']['acceptance_length']/raw['native_metrics']['acceptance_length']-1),
                    TPS_vs_RAW_pct=100*(window['tokens_per_second']/raw['tokens_per_second']-1),
                    performance_contaminated=window['performance_contaminated'],
                    clean_speed_comparison=all(row.get('speed_evidence',False) for row in [window,raw])))
    with (out/'FORMAL_RESULTS.csv').open('x') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    training=json.loads((out/'training/summary.json').read_text())
    (out/'RESULTS.json').write_text(json.dumps(dict(status='complete',training=training,formal_windows=rows,thinking=False,
        block_size=16,proposal_count=15,max_answer_tokens=2096,not_independent_final_test=True),ensure_ascii=False,indent=2)+'\n')
    lines=['nonthinking no-soft3：20,000条×3轮，正式native结果','block16=1锚点+15proposal；prompt+answer训练<=4096；评测回答max2096。','基底由本轮训练前128条的KL hidden负梯度方向未中心化二阶矩拟合；新头随机初始化。','C1=200条、C40=2000条，T0/T1，同卡RAW/修复；单轮开发评测。','正式窗口有其他进程或RAW参照被污染时，TPS比较不作干净提速证据。','全部epoch权重和完整Adam/FP32 master恢复状态保留。','']
    for row in rows:
        lines.append(f"C{row['C']} T{row['T']} {row['arm']}: AL={row['AL']:.6f}, TPS={row['TPS']:.3f}; 相对RAW AL {row['AL_vs_RAW_pct']:+.2f}%，TPS {row['TPS_vs_RAW_pct']:+.2f}%；污染={row['performance_contaminated']}")
    (out/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
