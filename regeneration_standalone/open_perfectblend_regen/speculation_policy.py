"""Audited serving overrides; target sampling and saved recipe stay immutable."""
import json
from pathlib import Path


def read_policy(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def selected_tokens(profile, gpu_index, native_tokens):
    policy = read_policy(profile.output_root / 'speculation_policy.json')
    worker = read_policy(profile.worker_dir(gpu_index) / 'speculation.json')
    tokens = worker.get('num_speculative_tokens', policy.get('num_speculative_tokens', native_tokens))
    if type(tokens) is not int or not 1 <= tokens <= native_tokens:
        raise ValueError(f'Draft length must be an integer in 1..{native_tokens}')
    return tokens


def with_tokens(command, tokens):
    result = list(command)
    position = result.index('--speculative-config') + 1
    config = json.loads(result[position])
    native = config['num_speculative_tokens']
    if type(tokens) is not int or not 1 <= tokens <= native:
        raise ValueError(f'Draft length must be an integer in 1..{native}')
    config['num_speculative_tokens'] = tokens
    result[position] = json.dumps(config)
    return result, config


def execution_record(server, contract, policy_path):
    """Require identical pinned models/method/sampling; only draft length varies."""
    command = server.get('command')
    if not command:  # Legacy/mock servers with no execution override.
        return None
    actual = json.loads(command[command.index('--speculative-config') + 1])
    anchored = contract['speculative_config']
    if {k:v for k,v in actual.items() if k != 'num_speculative_tokens'} != {
            k:v for k,v in anchored.items() if k != 'num_speculative_tokens'}:
        raise ValueError('Serving override changes the pinned draft, method or draft sampler')
    tokens = actual['num_speculative_tokens']
    if type(tokens) is not int or not 1 <= tokens <= anchored['num_speculative_tokens']:
        raise ValueError('Serving override exceeds the trained native draft length')
    return {'actual_speculative_config': actual,
            'anchored_speculative_config': anchored,
            'policy_path': str(policy_path),
            'target_sampling_unchanged': True,
            'override_scope': ['num_speculative_tokens']}


def choose_result(results, minimum_gain=.05):
    valid = [r for r in results if r['errors'] == 0 and r['processed'] == r['expected']]
    baseline = next(r for r in valid if r['num_speculative_tokens'] == 15)
    best = max(valid, key=lambda r:r['output_tokens_per_second'])
    gain = best['output_tokens_per_second'] / baseline['output_tokens_per_second'] - 1
    return (best if gain >= minimum_gain else baseline), gain
