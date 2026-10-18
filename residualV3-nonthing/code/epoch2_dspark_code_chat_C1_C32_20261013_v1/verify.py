"""Audit immutable sources, dataset identity, native contracts and formal evidence."""
from pathlib import Path
import json, random, sys

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE / 'native_runtime'))
from runtime.utils import dump, sha, now


def read(p): return json.loads(p.read_text())


def check(root, require_complete=True):
    plan = read(root / 'plan.json')
    pre = read(root / 'PREFLIGHT.json')
    assert pre['status'] == 'passed'
    assert all(sha(f) == digest for f, digest in plan['source_files'].items()), 'Source drift'
    assert all(sha(f) == digest for f, digest in pre['weight_and_dependency_checksums'].items()), 'Weight/dependency drift'
    source = Path(pre['parent'])
    assert all(sha(source / f) == sha(CODE / f) for f in pre['unchanged_native_runtime'])
    evidence = []
    for dataset, spec in plan['datasets'].items():
        template = root / dataset / 'native_protocol.template.json'
        assert sha(template) == spec['template_sha256']
        p = read(template)
        assert p['native_arms'] == ['DFLASH', 'FULL15_NOFILL']
        assert not p['cutoff_enabled'] and not p['auxiliary_predictors_enabled']
        assert p['body_models']['DFLASH'] != p['body_models']['FULL15_NOFILL']
        panel = Path(p['panel_paths']['final_C32'])
        rows = [json.loads(s) for s in panel.open(encoding='utf-8') if s.strip()]
        assert len(rows) == spec['count'] == len({r['id'] for r in rows})
        assert len({r['messages'][0]['content'] for r in rows}) == spec['unique_prompt_texts']
        assert all(len(r['messages']) == 1 and r.get('evaluated_turn_count',1) == 1 for r in rows)
        assert all(len(r['prompt_token_ids']) + 2048 <= p['native_max_model_len'] for r in rows)
        assert sha(panel) == p['panel_sha256']['final_C32'] == p['panel_sha256']['final_C1']
        assert p['architecture_by_arm']['FULL15_NOFILL']['depth'] == 3
        assert p['epoch_by_arm'] == {'DFLASH':2,'FULL15_NOFILL':2}
        assert p['kv_cache_gib_by_phase'] == {phase:{'C1':52,'C32':44} for phase in ['smoke','formal']}
        provenance=p['dataset_provenance']
        raw=Path(provenance['source_path'])
        assert sha(raw)==provenance['source_sha256']
        originals=[json.loads(line) for line in raw.open(encoding='utf-8') if line.strip()]
        indices=list(range(len(originals)))
        if len(indices)>spec['count']:
            random.Random(980406).shuffle(indices)
            indices=indices[:spec['count']]
        assert indices==provenance['source_rows_0based']
        assert all(row['messages'][0]['content']==originals[i]['turns'][0]
                   and row['source_line']==i+1 and row['sampling_seed']==980406+j
                   for j,(row,i) in enumerate(zip(rows,indices)))
        if not require_complete: continue
        for c in plan['concurrency']:
            folder = root / dataset / f'C{c}'
            formal, smoke = [read(folder / phase / 'summary.json') for phase in ('formal', 'formal_smoke')]
            assert formal['status'] == smoke['status'] == 'complete'
            assert formal['protocol']['gpu'] in plan['eligible_gpus']
            assert formal['protocol']['sources'] == plan['source_files']
            assert formal['protocol']['panel_sha256'] == sha(panel)
            assert formal['protocol']['prompt_count'] == spec['count']
            assert formal['protocol']['temperatures'] == [0, 1]
            assert formal['protocol']['max_tokens']==2048
            assert formal['protocol']['max_model_len']==p['native_max_model_len']
            assert formal['protocol']['kv_cache_memory_bytes']=={'1':52,'32':44}[str(c)]*(1<<30)
            assert formal['protocol']['kv_cache_memory_bytes']==smoke['protocol']['kv_cache_memory_bytes']
            assert formal['protocol']['training_protocol_sha256'] == smoke['protocol']['training_protocol_sha256'] == sha(folder / 'native_protocol.json')
            assert len(formal['windows']) == 4
            assert {(w['arm'], w['T']) for w in formal['windows']} == {(a, t) for a in plan['arms'] for t in [0, 1]}
            for w in formal['windows']:
                assert w['requests'] == spec['count']
                assert w['isolation_observed'] and w['speed_evidence'] and not w['performance_contaminated']
                for counter in ['execution_counters','target_execution_counters']:
                    k = w[counter]
                    assert k['replays'] > 0 and not any(k[x] for x in ['captures','eager','fallbacks'])
            requests = []
            for arm in plan['arms']:
                path = folder / 'formal/round_1' / arm / 'T0_requests.jsonl'
                requests.append([json.loads(s) for s in path.open(encoding='utf-8') if s.strip()])
            assert len(requests[0]) == len(requests[1]) == spec['count']
            unequal = [i for i, (a, b) in enumerate(zip(*requests))
                       if a['index'] != b['index'] or a['seed'] != b['seed']
                       or a['response']['choices'][0]['message']['content'] != b['response']['choices'][0]['message']['content']
                       or a['response']['usage']['completion_tokens'] != b['response']['usage']['completion_tokens']]
            assert not unequal, (dataset, c, 'T0 output mismatch', unequal)
            evidence.append(dict(dataset=dataset, concurrency=c, requests=spec['count'],
                                 T0_equal_outputs=True, isolation=True, Graph=True,
                                 formal_summary_sha256=sha(folder / 'formal/summary.json')))
    return dict(status='passed', source_files_checked=len(plan['source_files']),
                runtime_math_unchanged=True, weight_and_dependency_files_checked=len(pre['weight_and_dependency_checksums']),
                dataset_counts={d:s['count'] for d,s in plan['datasets'].items()},
                only_two_E2_arms=True, MT_Bench_first_turn_only=True, independent_quality_scores=False,
                require_complete=require_complete, formal_groups=evidence, utc=now())


def main():
    root = Path(sys.argv[1]).resolve()
    preflight = '--preflight' in sys.argv[2:]
    audit = check(root, not preflight)
    dump(root / ('READY_AUDIT.json' if preflight else 'FINAL_AUDIT.json'), audit)
    print(json.dumps(audit))


if __name__ == '__main__': main()
