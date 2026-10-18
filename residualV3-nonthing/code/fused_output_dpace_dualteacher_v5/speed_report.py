"""Summarize observed training supply times; shared GPU comparison is observational."""
import argparse,json,statistics
from pathlib import Path

def summarize(rows):
    keys=['seconds','hidden_wait_seconds','draft_seconds','head_seconds']
    return dict(count=len(rows),first_step=rows[0]['step'],last_step=rows[-1]['step'],
                mean={k:statistics.mean(r[k] for r in rows) for k in keys},median={k:statistics.median(r[k] for r in rows) for k in keys})
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[json.loads(line) for line in (a.run/'training_arms/CE_DPACE/training/metrics.jsonl').read_text().splitlines()]
    one=[r for r in rows if 'recovery_20261007_v4' in r.get('attempt','')][-20:]
    two=[r for r in rows if 'dualteacher_20261007_v5' in r.get('attempt','')]
    # Exclude the first two updates after process restart (loading/compilation/warmup).
    two=two[2:][-20:]
    assert len(one)==20 and len(two)>=10
    x,y=summarize(one),summarize(two)
    report=dict(single_teacher=x,dual_teacher=y,observed_step_speed_ratio=x['mean']['seconds']/y['mean']['seconds'],
                same_training_budget=True,training_only=True,shared_gpu=True,different_packs_and_external_load=True,
                limitation='before/after observation, not isolated proof of replica count causality or native inference throughput')
    (a.output/'SUPPLY_SPEED_COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
