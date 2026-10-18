"""Diagnostic-only CPU/CUDA trace, Graph event spans, and sampled proposal IDs."""
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch

ACTIVE = False


def marker():
    path = Path(os.environ["LOCAL_TRIAL"]) / "control.json"
    return json.loads(path.read_text()) if path.exists() else {"phase": "idle"}


def scope(name):
    return torch.profiler.record_function("diagnostic/" + name) if ACTIVE else nullcontext()


class Observer:
    def __init__(self):
        self.out = Path(os.environ["LOCAL_TRIAL"])
        self.prof = None
        self.profile_count = 0
        self.finished = set()
        self.event_count = {}
        self.id_count = {}

    def begin(self):
        global ACTIVE
        state = marker()
        key = (state["phase"], state.get("temperature"), state.get("index"))
        if key[0] != "profile" or key in self.finished:
            return
        if self.prof is None:
            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=False, profile_memory=False,
            )
            self.prof.start()
            self.profile_count = 0
            ACTIVE = True
        elif self.profile_count >= 4:
            ACTIVE = False
            self.prof.stop()
            stem = self.out / f"profile_T{key[1]}_p{key[2]}"
            self.prof.export_chrome_trace(str(stem) + ".json")
            rows = [
                dict(name=e.key, count=e.count, cpu_total_us=e.cpu_time_total,
                     self_cpu_us=e.self_cpu_time_total,
                     device_total_us=getattr(e, "device_time_total", 0),
                     self_device_us=getattr(e, "self_device_time_total", 0))
                for e in self.prof.key_averages()
            ]
            Path(str(stem) + "_ops.json").write_text(json.dumps(rows, indent=2) + "\n")
            Path(str(stem) + "_complete.json").write_text(
                json.dumps(dict(profiled_draft_calls=self.profile_count,
                                not_runtime_measurement=True)) + "\n"
            )
            self.prof = None
            self.finished.add(key)

    def after(self, model):
        state = marker()
        key = (state["phase"], state.get("temperature"), state.get("index"))
        if self.prof is not None and key[0] == "profile":
            self.profile_count += 1
        if key[0] in {"capture", "profile", "events"} and self.id_count.get(key, 0) < 4:
            row = dict(phase=key[0], temperature=key[1], index=key[2],
                       block=self.id_count.get(key, 0),
                       sampled_proposal_ids=model.last_proposal_ids.detach().cpu().tolist(),
                       note="actual sampled draft IDs before target verification")
            with (self.out / "proposal_ids.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            self.id_count[key] = row["block"] + 1
        if key[0] != "events" or self.event_count.get(key, 0) >= 4:
            return
        torch.cuda.synchronize(model.lm_head.weight.device)
        spans = {}
        for name, start, end in model.repair_graph.stage_events:
            spans[name] = spans.get(name, 0.0) + start.elapsed_time(end)
        row = dict(temperature=key[1], index=key[2],
                   block=self.event_count.get(key, 0), graph_stage_ms=spans,
                   raw_projection_ms=model.raw_events[0].elapsed_time(model.raw_events[1]),
                   input_copy_ms=model.copy_events[0].elapsed_time(model.copy_events[1]),
                   graph_replay_stream_ms=model.copy_events[1].elapsed_time(model.copy_events[2]),
                   note="diagnostic CUDA stream intervals with external Graph events and synchronization; not pure kernel time")
        with (self.out / "event_spans.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        self.event_count[key] = row["block"] + 1
