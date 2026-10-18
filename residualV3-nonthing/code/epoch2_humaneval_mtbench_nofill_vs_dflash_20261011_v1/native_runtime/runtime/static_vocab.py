from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
from .precision import require_bf16

def sha(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()

def validate_ids(ids: torch.Tensor, vocab_size: int, expected_size: int | None = None) -> None:
    if ids.ndim != 1 or ids.dtype != torch.long:
        raise TypeError('fixed M must be a 1D int64 tensor')
    if expected_size is not None and ids.numel() != expected_size:
        raise ValueError(f'M size {ids.numel()} != {expected_size}')
    if ids.numel() == 0 or ids.numel() > vocab_size:
        raise ValueError('invalid fixed M size')
    if bool((ids < 0).any()) or bool((ids >= vocab_size).any()):
        raise ValueError('fixed M token ID outside full vocabulary')
    if ids.numel() > 1 and not bool((ids[1:] > ids[:-1]).all()):
        raise ValueError('fixed M must be unique and strictly sorted by target token ID')

def load_ids(path: Path, expected_sha: str, vocab_size: int, expected_size: int) -> torch.Tensor:
    if sha(path) != expected_sha:
        raise RuntimeError('fixed M token ID file hash changed')
    values=json.loads(path.read_text())
    if not isinstance(values,list) or any(type(value) is not int for value in values):
        raise TypeError('fixed M JSON must contain integer target token IDs')
    ids=torch.tensor(values,dtype=torch.long)
    validate_ids(ids,vocab_size,expected_size)
    return ids

def make_static_basis(token_basis: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Construct one contiguous, shared B_M from the original full token basis."""
    require_bf16(token_basis=token_basis)
    validate_ids(ids,token_basis.shape[0])
    with torch.no_grad():
        return token_basis.index_select(0,ids.to(token_basis.device)).contiguous()
