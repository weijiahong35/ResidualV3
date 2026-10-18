"""Bounded completion-driven requests with consumption in exact input order."""
from threading import Condition


class OrderedRequests:
    def __init__(self, pool, fetch, concurrency=32, window=64):
        assert 0 < concurrency <= window
        self.pool, self.fetch = pool, fetch
        self.concurrency, self.window = concurrency, window
        self.cv = Condition()
        self.pending = {}
        self.running = self.submitted = self.consumed = 0
        self.closed = self.exhausted = False
        self.peak_running = self.peak_pending = 0
        self.source = iter(())

    def begin(self, rows):
        with self.cv:
            self.cv.wait_for(lambda: self.running == 0)
            assert not self.pending and not self.closed
            self.source = iter(rows)
            self.submitted = self.consumed = 0
            self.exhausted = False
            self._fill_locked()

    def _fill_locked(self):
        while (not self.closed and not self.exhausted and
               self.running < self.concurrency and len(self.pending) < self.window):
            row = next(self.source, None)
            if row is None:
                self.exhausted = True
                break
            index = self.submitted
            self.submitted += 1
            self.running += 1
            future = self.pool.submit(self.fetch, row)
            self.pending[index] = (row['input_ids'], future)
            self.peak_running = max(self.peak_running, self.running)
            self.peak_pending = max(self.peak_pending, len(self.pending))
            future.add_done_callback(self._completed)

    def _completed(self, future):
        with self.cv:
            self.running -= 1
            self._fill_locked()
            self.cv.notify_all()

    def get(self, row):
        with self.cv:
            ids, future = self.pending[self.consumed]
            assert ids == row['input_ids'], 'ordered hidden alignment'
        result = future.result()
        with self.cv:
            del self.pending[self.consumed]
            self.consumed += 1
            self._fill_locked()
        return result

    def close(self):
        with self.cv:
            self.closed = True
        # In-progress fetches still unlink their own temporary hidden files.
        self.pool.shutdown(wait=True, cancel_futures=True)
        with self.cv:
            self.pending.clear()
