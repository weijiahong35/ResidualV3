#!/usr/bin/env python3
"""Verify transient hidden files are deleted on success and validation failure."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from safetensors.torch import save_file
import torch

import Baseline_online_common as online


class FakeClient:
    def __init__(self, **_kwargs):
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id='target')]))


def main() -> None:
    real_openai = online.openai.OpenAI
    real_generate = online.generate_hidden_states
    online.openai.OpenAI = FakeClient
    try:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fetch = online.LiveHiddenFetcher('http://localhost:1/v1', root, 'target')
            ids = torch.tensor([1,2,3], dtype=torch.long)
            for fail in (False, True):
                def generate(_client, _model, _item, **_kwargs):
                    path = root / 'generated.safetensors'
                    payload = {'token_ids': ids.clone() if not fail else torch.tensor([9,9,9]),
                               'hidden_states': torch.zeros(3,6,4096,dtype=torch.bfloat16)}
                    save_file(payload,str(path))
                    return str(path)
                online.generate_hidden_states = generate
                if fail:
                    try:
                        fetch(ids)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('invalid target tokens accepted')
                else:
                    assert fetch(ids).shape == (3,6,4096)
                assert not list(root.iterdir()), 'transient hidden file was not removed'
            assert fetch.generated == fetch.deleted == 2
    finally:
        online.openai.OpenAI = real_openai
        online.generate_hidden_states = real_generate
    print('online generate/delete contract PASS', flush=True)


if __name__ == '__main__':
    main()
