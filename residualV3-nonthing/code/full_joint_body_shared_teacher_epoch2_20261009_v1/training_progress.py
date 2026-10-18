"""Separate a new epoch's local schedule from its inherited optimizer steps."""
import math


def lr_scale(step, total, fraction):
    assert 1 <= step <= total
    warmup = max(1, math.ceil(total * fraction))
    if step <= warmup:
        return step / warmup
    return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def resume_spec(initial, protocol, current_proof):
    checkpoint = initial['checkpoint']
    if initial['resume_kind'] == 'previous_epoch':
        assert initial['cursor'] == 0
        assert checkpoint['cursor'] == protocol['global_step_offset']
        return checkpoint, protocol['initial_checkpoint_proof'], checkpoint['cursor']
    assert initial['resume_kind'] == 'same_epoch'
    assert checkpoint['cursor'] == initial['cursor']
    assert checkpoint['epoch'] == protocol['training_epoch']
    assert checkpoint['global_step'] == protocol['global_step_offset'] + initial['cursor']
    return checkpoint, current_proof, initial['cursor']
