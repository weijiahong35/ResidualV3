"""CPU-only preparation of the author's exact six code/chat dataset selections."""
from pathlib import Path
import ast, copy, hashlib, json, random, shutil, sys
from transformers import AutoTokenizer

CODE = Path(__file__).resolve().parent
BASE = CODE.parents[2]
ROOT = BASE / 'residualV3-nonthing/results' / CODE.name
PARENT_CODE = BASE / 'residualV3-nonthing/code/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1'
PARENT_RESULTS = BASE / 'residualV3-nonthing/results/epoch2_three_domains_C32_nofill_vs_dflash_20261012_v1'
COMMIT = '005e03b81cec38b7da6399833d609ee89a2587f2'
DATA = BASE / 'regeneration_standalone/open_perfectblend_regen/eval_sources' / COMMIT
DATASETS = ['humaneval', 'mbpp', 'livecodebench', 'mt-bench', 'alpaca', 'arena-hard-v2']
sys.path.insert(0, str(CODE / 'native_runtime'))
from runtime.utils import dump, sha, now


def read(p):
    return json.loads(Path(p).read_text())


def main():
    assert not (ROOT / 'plan.json').exists(), 'Already frozen; prepare a new version instead'
    tree_path = BASE / 'residualV3-nonthing/research/training_recipe_review_20261006/deepseek-ai__DeepSpec_tree.json'
    tree = read(tree_path)
    assert tree['sha'] == COMMIT
    blobs = {x['path']: x['sha'] for x in tree['tree']}
    shutil.copy2(tree_path, ROOT / 'sources/upstream_tree.json')
    for rel in ['eval.py', 'deepspec/eval/base_evaluator.py', 'eval_datasets/convert_eval_datasets_to_jsonl.py']:
        data = (ROOT / 'sources' / rel.replace('/', '__')).read_bytes()
        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == blobs[rel]
    module = ast.parse((ROOT / 'sources/eval.py').read_text())
    tasks = dict(next(ast.literal_eval(n.value) for n in module.body
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TASKS' for t in n.targets)))
    tokenizer = AutoTokenizer.from_pretrained('/data/weijiahong/huggingface/models/Qwen3-8B', local_files_only=True)
    parent = read(PARENT_RESULTS / 'humaneval/native_protocol.template.json')
    source_files = {str(p): sha(p) for p in sorted(CODE.rglob('*.py'))}
    unchanged = [str(p.relative_to(CODE)) for p in sorted((CODE / 'native_runtime').rglob('*.py'))]
    assert all(sha(CODE / rel) == sha(PARENT_CODE / rel) for rel in unchanged)
    assert all((CODE / rel).stat().st_ino != (PARENT_CODE / rel).stat().st_ino for rel in unchanged)
    dependencies = read(PARENT_RESULTS / 'PREFLIGHT.json')['weight_and_dependency_checksums']
    assert all(sha(p) == digest for p, digest in dependencies.items())
    specs = {}
    audit = {}
    c1_capture = list(range(1, 17))
    c32_capture = sorted(set(range(1,17)) | set(range(24,513,8)))
    for number, ds in enumerate(DATASETS):
        snapshot = ROOT / 'sources' / f'{ds}.jsonl'
        shutil.copy2(DATA / f'{ds}.jsonl', snapshot)
        raw = snapshot.read_bytes()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        assert blob == blobs[f'eval_datasets/{ds}.jsonl']
        with snapshot.open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        selected = list(range(len(rows)))
        if len(rows) > tasks[ds]:
            random.Random(980406).shuffle(selected)
            selected = selected[:tasks[ds]]
        panel = []
        for index, source_index in enumerate(selected):
            row = rows[source_index]
            messages = [{'role': 'user', 'content': row['turns'][0]}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                add_generation_prompt=True, enable_thinking=False)
            assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)
            panel.append(dict(index=index, id=f'{ds}:{source_index}', source_line=source_index+1,
                 source_path=str(snapshot), messages=messages, prompt_token_ids=ids,
                 prompt_sha256=hashlib.sha256(messages[0]['content'].encode()).hexdigest(),
                 source_turn_count=len(row['turns']), evaluated_turn_count=1,
                 sampling_seed=980406+index, scope='Exact author source row and seeded subset; first turn'))
        context = 12288 if ds == 'arena-hard-v2' else 4096
        assert len(panel) == tasks[ds] and all(len(r['prompt_token_ids'])+2048 <= context for r in panel)
        panel_path = ROOT / ds / 'panels/dspark_selected.jsonl'
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.write_text(''.join(json.dumps(row, ensure_ascii=True)+'\n' for row in panel))
        unique = len({r['messages'][0]['content'] for r in panel})
        provenance = dict(commit=COMMIT, source_path=str(snapshot), source_sha256=sha(snapshot),
             upstream_git_blob=blob, source_url=f'https://github.com/deepseek-ai/DeepSpec/blob/{COMMIT}/eval_datasets/{ds}.jsonl',
             source_rows=len(rows), selected_rows=len(panel), selection_seed=980406,
             source_rows_0based=selected, unique_prompt_texts=unique, evaluated_turns_per_question=1)
        p = copy.deepcopy(parent)
        for key in ['C32_parent_protocol', 'source_files', 'server_port_base', 'allocation_policy']:
            p.pop(key, None)
        p.update(runtime=str(CODE / 'native_runtime'), source_files=source_files,
            dataset=ds, dataset_provenance=provenance,
            dataset_scope=f'{len(panel)} author-selected source rows; {unique} distinct first-turn prompt texts; no added repeats',
            scope=f'E2 {ds} C1/C32 T0/T1; exact DSpark dataset selection',
            gpu_by_concurrency={'C1':None,'C32':None}, gpu_uuid=None,
            prompts_per_concurrency={'C1':len(panel),'C32':len(panel)},
            native_max_tokens=2048, native_max_model_len=context, selection_seed=980406,
            panel_paths={'warmup':parent['panel_paths']['warmup'], 'final_C1':str(panel_path),'final_C32':str(panel_path)},
            panel_sha256={'warmup':parent['panel_sha256']['warmup'], 'final_C1':sha(panel_path),'final_C32':sha(panel_path)},
            kv_cache_gib_by_phase={phase:{'C1':52,'C32':44} for phase in ['smoke','formal']},
            target_capture_sizes_by_concurrency={'C1':c1_capture,'C32':c32_capture},
            requested_gpu_memory_utilization=.92, server_port_base=29600+number*10,
            implementation_iteration=CODE.name, parent_experiment=str(PARENT_RESULTS),
            implementation_scope='Native math unchanged; author data/seed/max-new-token alignment and Arena context capacity only',
            user_instruction='Only E2; redo code/chat using DSpark evaluation datasets',
            memory_allocation_policy='Fixed real KV: C1=52GiB, C32=44GiB in smoke and formal; utilization .92',
            allocation_policy='At most two owned workers; any host-empty GPU0-7 with free>=78000MiB',
            target_allocation_policy_unchanged=(context==4096),
            dspark_alignment=dict(official_source_files=True, pinned_commit=COMMIT,
                sample_counts=True, seeded_subsets_and_order=True, first_turn_only=True,
                nonthinking=True, max_new_tokens=2048, seed=980406, T1=True,
                confidence_scheduler_disabled=True, additional_T0=True,
                ours_proposals=15, dspark_proposals=7, ours_engine='native vLLM C1/C32',
                dspark_engine='offline Transformers batch1 per GPU',
                scope='Dataset and request protocol alignment, not DSpark weights/training/engine reproduction'), utc=now())
        template = ROOT / ds / 'native_protocol.template.json'
        dump(template, p)
        for c in [1,32]: (ROOT / ds / f'C{c}').mkdir(exist_ok=True)
        specs[ds] = dict(count=len(panel), unique_prompt_texts=unique, context_limit=context,
                        template_sha256=sha(template), domain='code' if number<3 else 'chat')
        audit[ds] = dict(provenance, max_prompt_tokens=max(len(r['prompt_token_ids']) for r in panel),
                        context_limit=context, panel_sha256=sha(panel_path), no_prompt_truncation=True)
    assert sum(s['count'] for s in specs.values()) == 2000
    dump(ROOT / 'DATASET_AUDIT.json', dict(status='passed', datasets=audit, source_commit=COMMIT,
         selection_algorithm='If source exceeds TASKS cap: random.Random(980406).shuffle, then first cap; otherwise original order',
         mbpp_source_duplicate_preserved=True, artificial_repetition=False, utc=now()))
    dump(ROOT / 'PREFLIGHT.json', dict(status='passed', weight_and_dependency_checksums=dependencies,
         unchanged_native_runtime=unchanged, parent=str(PARENT_CODE), copied_code_uses_independent_files=True,
         runtime_math_changed=False, evaluator_changes=['exact DSpark request seed metadata', 'configured Arena context limit', 'official source duplicates accurately labelled'], utc=now()))
    dump(ROOT / 'plan.json', dict(code=str(CODE), datasets=specs, arms=['DFLASH','FULL15_NOFILL'],
         epoch=2, head_depth=3, concurrency=[1,32], temperatures=[0,1], formal_windows_expected=48,
         formal_requests_expected=16000, max_new_tokens=2048, request_seed_base=980406,
         eligible_gpus=list(range(8)), maximum_simultaneous_owned_gpus=2,
         empty_gpu_free_mib=78000, user_disk_limit_bytes=600*(1<<30), source_files=source_files,
         created_utc=now(), previous_results_preserved=True, source_commit=COMMIT))
    control=ROOT/'parallel'
    dump(control/'allocation_policy.json',dict(eligible_gpus=list(range(8)),all_empty_GPUs_user_authorized=True,
         maximum_simultaneous_owned_gpus=2,foreign_processes_untouched=True))
    dispatch={str(CODE/f):sha(CODE/f) for f in ['parallel.py','update_report.py','verify.py']}
    dump(control/'DISPATCH_MANIFEST.json',dict(new_dispatch_sources=dispatch,root=str(ROOT),utc=now()))
    dump(ROOT/'status.json',dict(status='prepared',formal_windows_completed=0,formal_windows_expected=48,utc=now()))
    print(json.dumps({'status':'prepared','counts':{d:s['count'] for d,s in specs.items()},'formal_windows':48}))


if __name__ == '__main__': main()
