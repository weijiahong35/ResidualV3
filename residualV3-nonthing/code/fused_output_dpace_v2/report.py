"""Formal-only four-arm AL comparison; flag shared-card timing contamination."""
import argparse
import csv
import json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True)
    out=ap.parse_args().output; rows=[]
    for c in [1,40]:
        s=json.loads((out/f'C{c}/formal/summary.json').read_text())
        assert s['status']=='complete' and len(s['windows'])==8
        windows={(w['arm'],w['T']):w for w in s['windows']}
        for t in [0,1]:
            base=windows['OUTPUT',t]['native_metrics']['acceptance_length']
            raw=windows['RAW',t]['native_metrics']['acceptance_length']
            for arm in ['RAW','OUTPUT','CE_DPACE','CE_DPACE_OUTPUT']:
                w=windows[arm,t]; al=w['native_metrics']['acceptance_length']
                rows.append(dict(C=c,T=t,arm=arm,AL=al,AL_vs_OUTPUT_pct=100*(al/base-1),
                     AL_vs_RAW_pct=100*(al/raw-1),TPS=w['tokens_per_second'],
                     performance_contaminated=w['performance_contaminated'],speed_evidence=w['speed_evidence']))
    with (out/'FORMAL_RESULTS.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'RESULTS.json').write_text(json.dumps(dict(status='complete',rows=rows,
         development_panel=True,OUTPUT_reuses_completed_weight=True,
         numerical_implementation_difference='old OUTPUT chunk64 vs new validated fused loss chunk1536'),indent=2)+'\n')
    lines=['三种损失：相同nonthinking20k×3、no-soft三层冻结主体修复头。',
           'OUTPUT沿用已完成权重；CE_DPACE与CE_DPACE_OUTPUT从相同随机初始化训练。',
           '65536输入token/包、整包1536anchors、894更新；完整词表目标argmax CE，D-PACE alpha0.5；联合项各系数1。',
           '新组使用融合损失与1536整块，旧OUTPUT用64分块；数学预算一致，浮点累积实现不同。',
           'native修复仍递推采样，15 proposals；C1=200/C40=2000，T0/T1，最多2096输出token。',
           'AL含一个bonus；共享GPU性能窗口标记污染，smoke不作测速；开发面板不等于独立最终测试。','']
    lines.extend(f"C{r['C']} T{r['T']} {r['arm']}: AL={r['AL']:.6f}; 相对OUTPUT {r['AL_vs_OUTPUT_pct']:+.2f}%; 性能污染={r['performance_contaminated']}" for r in rows)
    (out/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
