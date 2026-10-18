"""CPU-only validation of observed native Graph batch-size coverage."""
from __future__ import annotations


def coverage(records: list[dict], capacity: int) -> dict:
    if not 1 <= capacity <= 30:
        raise ValueError('capacity must be in 1..30')
    expected=set(range(1,capacity+1))
    observed={int(row['batch_size']) for row in records}
    pointers={int(row['q_data_ptr']) for row in records}
    invalid=observed-expected
    if invalid:
        raise ValueError(f'captured batch size outside configured capacity: {sorted(invalid)}')
    return dict(expected_batch_sizes=sorted(expected),observed_batch_sizes=sorted(observed),
                missing_batch_sizes=sorted(expected-observed),q_storage_pointers=sorted(pointers),
                passed=observed==expected and len(pointers)==1)
