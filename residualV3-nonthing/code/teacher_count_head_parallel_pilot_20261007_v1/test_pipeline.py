"""Concurrency, order, cancellation, epoch and CPU packing gates."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
import ast
from pathlib import Path
import torch
from ordered_scheduler import OrderedRequests
from pipeline import allocate_host, fill_host


def main():
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    rows = [dict(input_ids=[i]) for i in range(40)]
    entered, release = Event(), Event()
    lock, seen = Lock(), []
    def fetch(row):
        i = row['input_ids'][0]
        if i == 0:
            entered.set()
            assert release.wait(10)
        with lock:
            seen.append(i)
        return i
    ordered = OrderedRequests(ThreadPoolExecutor(max_workers=2), fetch, 2, 4)
    ordered.begin(rows)
    assert entered.wait(2)
    with ordered.cv:
        assert ordered.cv.wait_for(lambda: 3 in seen, timeout=2)
    assert 0 not in seen, 'completed requests must refill ahead of slow first result'
    release.set()
    assert [ordered.get(row) for row in rows] == list(range(40))
    ordered.begin(rows[1:3])
    assert [ordered.get(row) for row in rows[1:3]] == [1, 2]
    assert ordered.peak_running <= 2 and ordered.peak_pending <= 4
    ordered.close()
    def failing(row):
        if row['input_ids'][0] == 2:
            raise ValueError('deliberate fetch error')
        return row['input_ids'][0]
    bad = OrderedRequests(ThreadPoolExecutor(max_workers=2), failing, 2, 4)
    bad.begin(rows)
    assert bad.get(rows[0]) == 0 and bad.get(rows[1]) == 1
    try:
        bad.get(rows[2])
    except ValueError:
        pass
    else:
        raise AssertionError('fetch error was hidden')
    bad.close()
    torch.set_num_threads(2)
    documents = [dict(input_ids=list(range(n)), loss_mask=[True]*n) for n in [31, 23, 37]]
    chunks = [torch.randn(len(row['input_ids']), 6, 8).bfloat16() for row in documents]
    host = allocate_host(documents, 128, width=8, pinned=False)
    iterator = iter(chunks)
    fill_host(host, documents, lambda row: next(iterator))
    expected = torch.cat(chunks)
    assert torch.equal(host.aux[0, :91], expected[:, :-1].flatten(1))
    assert torch.equal(host.final, expected[:, -1])
    assert not host.aux[0, 91:].count_nonzero()
    expected_eligible = [i for i in range(15)] + [31+i for i in range(7)] + [54+i for i in range(21)]
    assert host.eligible.tolist() == expected_eligible
    print('PASS: completion refill, FIFO order, bounded requests, epoch reset, error/close, CPU pack/padding/eligibility')


if __name__ == '__main__':
    main()
