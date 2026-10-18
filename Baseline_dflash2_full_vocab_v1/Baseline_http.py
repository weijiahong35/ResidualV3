from __future__ import annotations
import json,re,urllib.request,urllib.error
from collections import defaultdict
from typing import Any

_PROM_SIMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+([0-9.eE+-]+)$")
_PROM_LABELED_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]+)\}\s+([0-9.eE+-]+)$"
)
_SPEC_PREFIX = "vllm:spec_decode"

def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120.0) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def _http_text(url: str, timeout: float = 30.0) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _parse_spec_metrics(text: str) -> tuple[dict[str, float], dict[int, float]]:
    counters: dict[str, float] = defaultdict(float)
    per_pos: dict[int, float] = defaultdict(float)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM_SIMPLE_RE.match(line) or _PROM_LABELED_RE.match(line)
        if not m:
            continue
        groups = m.groups()
        name = groups[0].replace("_total", "")
        if _SPEC_PREFIX not in name:
            continue
        value = float(groups[-1])
        labels = groups[1] if len(groups) == 3 else None
        if "per_pos" in name and labels:
            pm = re.search(r'position="(\d+)"', labels)
            if pm:
                per_pos[int(pm.group(1))] += value
        elif "per_pos" not in name:
            counters[name] += value
    return dict(counters), dict(per_pos)


def _metrics_delta(before_text: str, after_text: str) -> dict[str, Any]:
    before, before_pos = _parse_spec_metrics(before_text)
    after, after_pos = _parse_spec_metrics(after_text)

    def d(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    drafts = d("vllm:spec_decode_num_drafts")
    draft_tokens = d("vllm:spec_decode_num_draft_tokens")
    accepted = d("vllm:spec_decode_num_accepted_tokens")
    positions = sorted(set(before_pos) | set(after_pos))
    pos_delta = {p: after_pos.get(p, 0.0) - before_pos.get(p, 0.0) for p in positions}

    result: dict[str, Any] = {
        "num_drafts": drafts,
        "num_draft_tokens": draft_tokens,
        "num_accepted_tokens": accepted,
        "acceptance_length": (1.0 + accepted / drafts) if drafts > 0 else 0.0,
    }
    for p in positions:
        result[f"acceptance_at_pos_{p}"] = pos_delta[p] / drafts if drafts > 0 else 0.0
    return result

