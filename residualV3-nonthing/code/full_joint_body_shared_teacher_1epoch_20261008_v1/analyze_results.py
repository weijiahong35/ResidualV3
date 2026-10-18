"""Audit actual paired completion throughput, including bounded shared supply."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();r=a.root
    p=json.loads((r/'protocol.json').read_text());arms=['joint','body']
    rows={arm:[json.loads(s) for s in (r/arm/'metrics.jsonl').read_text().splitlines()] for arm in arms}
    assert all(len(v)==p['pack_count'] for v in rows.values())
    paired=all(x['anchor_sha256']==y['anchor_sha256'] and x['packed_ids_sha256']==y['packed_ids_sha256'] and x['input_tokens']==y['input_tokens'] for x,y in zip(rows['joint'],rows['body']));assert paired
    cut=p['warmup_packs'];done=[max(x['finished_unix_time'],y['finished_unix_time']) for x,y in zip(rows['joint'],rows['body'])]
    elapsed=done[-1]-done[cut-1];steps=len(done)-cut;cycle=elapsed/steps
    producer=[json.loads(s) for s in (r/'producer_metrics.jsonl').read_text().splitlines()]
    cleanup=json.loads((r/'hidden_cleanup.json').read_text());assert cleanup['requests']==cleanup['deleted']==cleanup['expected_requests']
    weights=all((r/arm/'joint_pilot_weights.safetensors').exists() and (r/arm/'joint_pilot_full_state.pt').exists() for arm in arms)
    frozen=json.loads((r/'driver_ready.json').read_text())['source_files'];assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in frozen.items())
    means={arm:json.loads((r/arm/'summary.json').read_text()) for arm in arms}
    est=json.loads((r.parent/'joint_teacher1_vs2_timing_20261007_v2/FULL_DATA_TIME_ESTIMATE.json').read_text())
    result=dict(status='complete',layout=p['layout'],measured_packs=steps,paired_inputs_and_anchors_match=paired,hidden_fetched_once_verified=True,hidden_requests=cleanup['requests'],shared_queue_capacity=2,source_sha_match=True,weights_and_complete_states_saved=weights,bf16_weights_saved=all((r/arm/'joint_pilot_weights.safetensors').exists() for arm in arms),concurrent_measured_seconds=elapsed,seconds_per_paired_pack=cycle,total_input_tokens_per_measured_arm=sum(x['input_tokens'] for x in rows['joint'][cut:]),joint=means['joint'],body=means['body'],producer_means={k:statistics.mean(v[k] for v in producer[cut:]) for k in ['allocate_shared_seconds','fetch_fill_seconds','producer_wall_seconds']},full_data_pack_count=est['pack_count'],full_data_one_epoch_compute_hours=est['pack_count']*cycle/3600,scope='One concurrent 24-pack pilot, first8 excluded; shared queue wait includes teacher supply and backpressure from the other learner. Pure-DFlash standalone speed is not measured here.')
    (r/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['shared_queue_wait_seconds','host_pin_seconds','h2d_wall_seconds','body_forward_wall','projection_wall','head_forward_loss_wall','head_backward_wall','body_backward_wall','clip_wall','optimizer_wall','total_update_seconds']
    labels=['共享队列等待','本地固定内存拷贝','H2D与冻结target norm','DFlash前向','raw/gold LM投影','修复头/损失前向','logits/损失/头反向','DFlash反向','梯度裁剪','参数更新与回写','learner整步总时间']
    table=[];lines=['一张teacher共享hidden，两张卡分别训练DFlash+头、纯DFlash。','每组24包，每包65536-token预算、1536anchors整块、15个预测位置、一包一次更新。','各自独立权重和优化器；同一输入只请求一次teacher；hidden读取复制后删掉传输文件。','','过程 | DFlash+头 秒 | 纯DFlash 秒']
    for field,label in zip(fields,labels):
        x,y=[means[arm]['means'][field] for arm in arms];table.append(dict(stage=field,joint_seconds=x,body_seconds=y));lines.append(f'{label} | {x:.4f} | {y:.4f}')
    with (r/'timing_comparison.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=table[0].keys());w.writeheader();w.writerows(table)
    lines.extend(['',f'两组共同完成1包的端到端周期：{cycle:.4f}秒；后16包共同完成耗时：{elapsed:.3f}秒。',f'按全数据{est["pack_count"]}包外推1epoch：{result["full_data_one_epoch_compute_hours"]:.2f}小时，未加入定期保存、资源变化与等待空卡。',f'共享producer后16包平均分配/填充耗时：{result["producer_means"]}。',f'目标hidden请求/读取删除次数：{cleanup["requests"]}/{cleanup["deleted"]}，匹配一次遍历24包的输入条数。','共享队列等待不能全部归因于teacher慢：两槽上限也会让较快learner等待较慢learner释放。','本轮纯DFlash的整步速度是共享并发条件下的速度，不能当成它独占teacher时的速度。','两组权重均保留；本轮仅保存BF16权重，前一轮完整optimizer/master/RNG断点保留。' if not p.get('save_optimizer_state',True) else '两组实际GPU参数和主权重均更新；保存权重、optimizer、master和RNG。','仅为训练吞吐pilot，未进行模型质量或native接受长度评测。'])
    (r/'RESULTS_CN.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':main()
