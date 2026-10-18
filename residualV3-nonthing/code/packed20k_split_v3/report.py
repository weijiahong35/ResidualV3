"""Compare formal three-arm windows; explicitly report changed supervision."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    rows = []
    original, packed = 'N_per_sequence_20k', 'N_packed65536_a1536_20k'
    for concurrency in [1, 40]:
        summary = json.loads((out / f'C{concurrency}/formal/summary.json').read_text())
        assert summary['status'] == 'complete' and len(summary['windows']) == 6
        windows = {(row['arm'], row['T']): row for row in summary['windows']}
        for temperature in [0, 1]:
            raw, baseline = windows['RAW', temperature], windows[original, temperature]
            for arm in ['RAW', original, packed]:
                window = windows[arm, temperature]
                rows.append(dict(C=concurrency, T=temperature, arm=arm,
                    AL=window['native_metrics']['acceptance_length'], TPS=window['tokens_per_second'],
                    AL_vs_RAW_pct=100 * (window['native_metrics']['acceptance_length'] / raw['native_metrics']['acceptance_length'] - 1),
                    TPS_vs_RAW_pct=100 * (window['tokens_per_second'] / raw['tokens_per_second'] - 1),
                    AL_vs_original_pct=100 * (window['native_metrics']['acceptance_length'] / baseline['native_metrics']['acceptance_length'] - 1),
                    TPS_vs_original_pct=100 * (window['tokens_per_second'] / baseline['tokens_per_second'] - 1),
                    performance_contaminated=window['performance_contaminated'],
                    clean_RAW_comparison=all(item['speed_evidence'] for item in [raw, window]),
                    clean_original_comparison=all(item['speed_evidence'] for item in [baseline, window])))
    with (out / 'FORMAL_RESULTS.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plan = json.loads((out / 'plan.json').read_text())
    training = {}
    for arm, root in [(original, Path(plan['baseline_run'])), (packed, out)]:
        records = [json.loads(line) for line in (root / 'training/metrics.jsonl').read_text().splitlines()]
        training[arm] = dict(updates=len(records), sampled_anchors=sum(row['anchors'] for row in records),
            input_tokens=sum(row.get('input_tokens', row.get('example_tokens', 0)) for row in records),
            sum_logged_update_seconds=sum(row['seconds'] for row in records),
            scope='update timers, including hidden wait; exclude startup, basis fitting and checkpoint save overhead')
    (out / 'RESULTS.json').write_text(json.dumps(dict(status='complete', windows=rows, training=training,
        comparison='same data, epochs, models, basis, initial head; changed sampling budget, sample weighting and optimizer cadence',
        not_independent_final_test=True), ensure_ascii=False, indent=2) + '\n')
    lines = ['20k×3轮：逐条训练与65536输入token/整包1536 anchor训练的正式对比。',
        '两组均冻结相同target/draft、相同no-soft三层头、相同basis和初始头。',
        '采样预算、样本权重和更新频率同时变化，不能称纯打包工程消融。',
        'RAW/旧逐条/新打包：C1=200、C40=2000，各T0/T1，max2096；block16=已知锚点+15 proposal。',
        '其他进程污染的窗口有明确标记；smoke不计作性能结果。开发面板不等于独立最终测试。', '']
    for row in rows:
        lines.append(f"C{row['C']} T{row['T']} {row['arm']}: AL={row['AL']:.6f}, TPS={row['TPS']:.3f}; 相对旧逐条 AL {row['AL_vs_original_pct']:+.2f}%, TPS {row['TPS_vs_original_pct']:+.2f}%; 污染={row['performance_contaminated']}")
    lines.extend(['', json.dumps(training, ensure_ascii=False)])
    (out / 'RESULTS_CN.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
