"""Diagnostic Graph event nodes. These spans are not uninstrumented latency."""
import os
from contextlib import contextmanager
import torch
CURRENT=None
@contextmanager
def capture_events(events):
    global CURRENT
    previous=CURRENT;CURRENT=events if os.environ.get('DIAG_KIND')=='profile' else None
    try:yield
    finally:CURRENT=previous
@contextmanager
def stage(name):
    if CURRENT is None:
        yield;return
    start=torch.cuda.Event(enable_timing=True,external=True);end=torch.cuda.Event(enable_timing=True,external=True)
    CURRENT.append((name,start,end));start.record()
    try:yield
    finally:end.record()
