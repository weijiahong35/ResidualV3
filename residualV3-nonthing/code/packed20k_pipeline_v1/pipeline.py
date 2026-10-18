"""One current pack plus one next pack, with online RAM-only hidden transfer."""
from queue import Empty, Queue
from threading import BoundedSemaphore, Event, Lock, Thread


class PackPipeline:
    def __init__(self, provider, train, packs, depth=2):
        assert depth == 2
        self.provider, self.train, self.packs = provider, train, packs
        self.queue = Queue(maxsize=depth)
        self.slots = BoundedSemaphore(depth)
        self.cancel = Event()
        self.finished = Event()
        self.lock = Lock()
        self.error = None
        self.resident = self.peak_resident = 0
        provider.begin_epoch([train[index] for indices in packs for index in indices])
        self.thread = Thread(target=self._produce, name='next-pack-hidden', daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            for indices in self.packs:
                while not self.cancel.is_set():
                    if self.slots.acquire(timeout=.1):
                        break
                else:
                    return
                with self.lock:
                    self.resident += 1
                    self.peak_resident = max(self.peak_resident, self.resident)
                rows = [self.train[index] for index in indices]
                chunks = []
                for row in rows:
                    if self.cancel.is_set():
                        return
                    chunks.append(self.provider(row))
                if self.cancel.is_set():
                    return
                self.queue.put((indices, rows, chunks))
        except BaseException as error:
            self.error = error
        finally:
            self.finished.set()

    def get(self):
        while True:
            try:
                return self.queue.get(timeout=.1)
            except Empty:
                if self.finished.is_set():
                    if self.error is not None:
                        raise self.error
                    raise RuntimeError('Next-pack pipeline exhausted unexpectedly')

    def release(self):
        with self.lock:
            self.resident -= 1
        self.slots.release()

    def close(self):
        self.cancel.set()
        self.thread.join()
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                break

