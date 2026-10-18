"""Reuse complete valid conversations and their valid prefixes under an 8K cap."""
from collections import OrderedDict
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', CUDA_VISIBLE_DEVICES='')
spec = importlib.util.spec_from_file_location('pool_launcher', HERE / 'run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
import regenerate
from regenlib.checkpoint import ConversationCheckpoint, durable_json, repair_output_tail
from pool import DEFAULT_POOL, sha256_file, write_json

validation_spec = importlib.util.spec_from_file_location('residual_validation',
    ROOT.parent / 'residualV3_main_v1/runtime/data.py')
validation = importlib.util.module_from_spec(validation_spec)
validation_spec.loader.exec_module(validation)


def valid_prefix(samples):
    kept = []
    for sample in samples:
        if not validation.validate_example(sample):
            break
        kept.append(copy.deepcopy(sample))
    return kept


def progress_from_samples(turns, samples, decoder, assistant_messages=None):
    """Rebuild only the prefix before the first incompatible response."""
    prefix = []
    answered = 0
    next_turn = 0
    for index, turn in enumerate(turns):
        if answered == len(samples):
            break
        prefix.append(copy.deepcopy(turn))
        next_turn = index + 1
        if turn['role'] != 'user':
            continue
        sample = samples[answered]
        start = sample['loss_mask'].index(1)
        tokens = sample['input_ids'][start:]
        if tokens[-1:] == [151645]:
            tokens = tokens[:-1]
        message = copy.deepcopy(assistant_messages[answered]) if assistant_messages is not None else {
            'role': 'assistant', 'content': decoder(tokens)}
        prefix.append(message)
        answered += 1
    if answered != len(samples):
        raise ValueError('Saved response count exceeds source user turns')
    return {'samples': samples, 'prefix': prefix, 'next_turn': next_turn,
            'active_turn': False, 'tool_results': [], 'truncated': False}


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(launcher.backend.MODELS / 'Qwen3-8B'), local_files_only=True)
    previous = sys.argv
    try:
        sys.argv = ['run.py', '--pool-dir', str(DEFAULT_POOL), '--gpu', '1']
        args = launcher.parse_args()
        runtime, plan = launcher.make_plan(args)
    finally:
        sys.argv = previous
    destination = Path(plan['generation_root'])
    if destination.exists():
        raise RuntimeError('New generation root already exists; refusing to overwrite: ' + str(destination))
    staging = destination.with_name(destination.name + f'.migrating-{os.getpid()}')
    staging.mkdir()
    old_root = DEFAULT_POOL / 'regenerated'
    report = {'state': 'migrating', 'source': str(old_root), 'destination': str(destination),
              'training_length': 8192, 'minimum_supervised_tokens': 8,
              'whole_responses_preserved': True, 'reused_complete_conversations': 0,
              'reused_training_rows': 0, 'reused_partial_conversation_checkpoints': 0,
              'reused_prefix_rows': 0, 'conversations_requiring_new_budget': 0,
              'old_checkpoint_content_checks': 0, 'source_file_hashes': {}}
    report_path = HERE / 'gpu1_pilot_20261004/ctx8192_migration.json'
    write_json(report_path, report)
    root = staging / plan['thinking']
    root.mkdir()
    durable_json(root / 'generation_contract.json', plan['contract'])
    reused_ids = set()
    for job in plan['jobs']:
        source = Path(job['source'])
        with source.open() as handle:
            raw = {row['id']: (index, row) for index, line in enumerate(handle)
                   if (row := json.loads(line))}
        output = staging / Path(job['output']).relative_to(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        old_output = old_root / plan['thinking'] / job['split'] / output.name
        grouped = OrderedDict()
        if old_output.exists():
            repair_output_tail(old_output)
            report['source_file_hashes'][str(old_output)] = sha256_file(old_output)
            with old_output.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    grouped.setdefault(row['primary_id'], []).append(row)
        job_kept = 0
        with output.open('w') as handle:
            for primary_id, samples in grouped.items():
                if primary_id not in raw:
                    raise ValueError('Foreign conversation in legacy shard: ' + primary_id)
                kept = valid_prefix(samples)
                for sample in kept:
                    sample['metadata']['length_migration'] = {'source_file': str(old_output),
                        'original_context_limit': 16384, 'accepted_training_limit': 8192,
                        'migration_report': str(report_path), 'response_unchanged': True}
                if len(kept) == len(samples):
                    for sample in kept:
                        handle.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    reused_ids.add(primary_id)
                    job_kept += 1
                    report['reused_complete_conversations'] += 1
                    report['reused_training_rows'] += len(kept)
                else:
                    report['conversations_requiring_new_budget'] += 1
                    if kept:
                        index, row = raw[primary_id]
                        normalized, turns, tools = regenerate.prepare_row(row, regenerate.DATASET_CONFIGS['open-perfectblend'])
                        item = {'turns': turns, 'tool_results': tools}
                        client_args = type('Settings', (), dict(model=str(runtime.model), sampling_params=plan['contract']['sampling_params'],
                            max_tokens=8192, max_model_len=8192, minimum_valid_tokens=8, skip_context_overflow=True))()
                        checkpoint = ConversationCheckpoint(str(destination / Path(job['output']).relative_to(destination)) + '.checkpoints',
                            primary_id, regenerate.checkpoint_recipe(item, client_args))
                        # Write to staging, retaining the eventual checkpoint filename.
                        checkpoint.path = staging / checkpoint.path.relative_to(destination)
                        checkpoint.save(progress_from_samples(turns, kept,
                            lambda tokens: tokenizer.decode(tokens, skip_special_tokens=False)))
                        report['reused_partial_conversation_checkpoints'] += 1
                        report['reused_prefix_rows'] += len(kept)
            handle.flush()
            os.fsync(handle.fileno())
        old_checkpoints = Path(str(old_output) + '.checkpoints')
        for file in old_checkpoints.glob('*.json'):
            record = json.loads(file.read_text())
            primary_id = record['primary_id']
            if primary_id in reused_ids:
                continue
            if primary_id not in raw:
                raise ValueError('Foreign checkpoint: ' + primary_id)
            progress = record['progress']
            assistants = [message for message in progress['prefix'] if message['role'] == 'assistant']
            for sample, message in zip(progress['samples'], assistants, strict=True):
                tokens = sample['input_ids'][sample['loss_mask'].index(1):]
                if tokens[-1:] == [151645]:
                    tokens = tokens[:-1]
                if tokenizer.decode(tokens, skip_special_tokens=False) != message['content']:
                    raise ValueError('Cannot reproduce legacy assistant history exactly: ' + primary_id)
                report['old_checkpoint_content_checks'] += 1
            kept = valid_prefix(progress['samples'])
            if not kept:
                continue
            for sample in kept:
                sample['metadata']['length_migration'] = {'source_checkpoint': str(file),
                    'original_context_limit': 16384, 'accepted_training_limit': 8192,
                    'migration_report': str(report_path), 'response_unchanged': True}
            index, row = raw[primary_id]
            normalized, turns, tools = regenerate.prepare_row(row, regenerate.DATASET_CONFIGS['open-perfectblend'])
            item = {'turns': turns, 'tool_results': tools}
            client_args = type('Settings', (), dict(model=str(runtime.model), sampling_params=plan['contract']['sampling_params'],
                max_tokens=8192, max_model_len=8192, minimum_valid_tokens=8, skip_context_overflow=True))()
            checkpoint = ConversationCheckpoint(str(output) + '.checkpoints', primary_id,
                                                regenerate.checkpoint_recipe(item, client_args))
            checkpoint.save(progress_from_samples(turns, kept, None, assistants))
            report['reused_partial_conversation_checkpoints'] += 1
            report['reused_prefix_rows'] += len(kept)
        if job_kept:
            durable_json(output.with_suffix('.status.json'), {'state': 'partial',
                'completed_conversations': job_kept, 'skipped_conversations': 0,
                'expected_conversations': job['full_shard_conversations'], 'migration_report': str(report_path)})
    durable_json(root / 'training_data_paths.json', {split: [str(destination / p.relative_to(staging))
        for p in sorted((root / split).glob('shard-*.jsonl')) if p.stat().st_size]
        for split in ['train', 'validation']})
    staging.rename(destination)
    report['state'] = 'complete'
    report['generation_contract'] = str(destination / plan['thinking'] / 'generation_contract.json')
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
