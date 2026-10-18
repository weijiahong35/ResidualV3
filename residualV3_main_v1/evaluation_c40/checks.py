"""CPU-only validation of observed native Graph batch-size coverage."""
from __future__ import annotations


def coverage(records: list[dict], capacity: int) -> dict:
    if not 1 <= capacity <= 40:
        raise ValueError('capacity must be in 1..40')
    expected=set(range(1,capacity+1))
    observed={int(row['batch_size']) for row in records}
    pointers={int(row['q_data_ptr']) for row in records}
    invalid=observed-expected
    if invalid:
        raise ValueError(f'captured batch size outside configured capacity: {sorted(invalid)}')
    return dict(expected_batch_sizes=sorted(expected),observed_batch_sizes=sorted(observed),
                missing_batch_sizes=sorted(expected-observed),q_storage_pointers=sorted(pointers),
                passed=observed==expected and len(pointers)==1)


def peak_inflight(rows):
    events=[]
    for row in rows:
        if row['end_unix_ns']<=row['start_unix_ns']:raise ValueError('invalid request interval')
        events.extend(((row['start_unix_ns'],1),(row['end_unix_ns'],-1)))
    current=peak=0
    for _,change in sorted(events):
        current+=change;peak=max(peak,current)
    return peak
