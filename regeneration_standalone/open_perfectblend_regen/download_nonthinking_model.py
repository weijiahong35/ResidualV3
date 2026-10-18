"""Resumable ranged download of the pinned non-thinking draft into the HF cache."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from pool import write_json

REPO = 'RedHatAI/Qwen3-8B-speculator.dflash'
REVISION = '1a11b170eb65c8a62c80ecd01dfe22a5907298e6'
SHA256 = 'afda76166ccd6a2d3ef55b214889bd045be3b8fc41b41be330d06d25059f88e8'
SIZE = 4586578776
CHUNK = 16 * 2**20
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error('Download workers must be between 1 and 64')
    from huggingface_hub.constants import HF_HUB_CACHE
    cache = Path(HF_HUB_CACHE) / ('models--' + REPO.replace('/', '--'))
    blob = cache / 'blobs' / SHA256
    snapshot = cache / 'snapshots' / REVISION
    parts = cache / 'download-parts' / SHA256
    parts.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors?download=true'
    chunks = (SIZE + CHUNK - 1) // CHUNK
    initial_bytes = sum(path.stat().st_size for path in parts.glob('*.part'))
    started = time.monotonic()

    def download(index):
        begin, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
        path = parts / f'{index:04d}.part'
        for attempt in range(12):
            offset = path.stat().st_size if path.exists() else 0
            if offset == end - begin + 1:
                return index
            if offset > end - begin + 1:
                raise ValueError('Oversized download part: ' + str(path))
            try:
                request = urllib.request.Request(url,
                    headers={'Range': f'bytes={begin + offset}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f'bytes {begin + offset}-{end}/{SIZE}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Unexpected range response: ' + str(response.headers.get('Content-Range')))
                    with path.open('ab') as handle:
                        while True:
                            data = response.read(2**20)
                            if not data:
                                break
                            handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                if path.stat().st_size == end - begin + 1:
                    return index
            except Exception as error:
                print(json.dumps({'retry_part': index, 'attempt': attempt + 1, 'error': repr(error)}), flush=True)
                time.sleep(min(20, 2 + attempt * 2))
        raise RuntimeError(f'Download retries exhausted for part {index}')

    if not blob.exists():
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(download, index) for index in range(chunks)]
            for future in as_completed(futures):
                index = future.result()
                downloaded = sum(path.stat().st_size for path in parts.glob('*.part'))
                elapsed = time.monotonic() - started
                value = {'state': 'downloading', 'revision': REVISION, 'downloaded_bytes': downloaded,
                    'total_bytes': SIZE, 'percent': round(downloaded / SIZE * 100, 2),
                    'elapsed_seconds': round(elapsed, 1),
                    'mean_mib_per_second': round((downloaded - initial_bytes) / elapsed / 2**20, 2)}
                write_json(HERE / 'nonthinking_model_download_progress.json', value)
                print(json.dumps(value), flush=True)
        partial = blob.with_suffix('.assembled')
        digest = hashlib.sha256()
        with partial.open('wb') as output:
            for index in range(chunks):
                with (parts / f'{index:04d}.part').open('rb') as source:
                    while data := source.read(8 * 2**20):
                        digest.update(data)
                        output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if partial.stat().st_size != SIZE or digest.hexdigest() != SHA256:
            raise ValueError('Pinned model download size/SHA256 mismatch')
        os.replace(partial, blob)
    else:
        assert blob.stat().st_size == SIZE
        digest = hashlib.sha256()
        with blob.open('rb') as source:
            while data := source.read(8 * 2**20):
                digest.update(data)
        assert digest.hexdigest() == SHA256
    link = snapshot / 'model.safetensors'
    if not link.exists():
        link.symlink_to(Path('../../blobs') / SHA256)
    model_link = Path('/data/weijiahong/huggingface/models/Qwen3-8B-speculator.dflash-nonthinking-20260916')
    if not model_link.exists():
        model_link.symlink_to(snapshot, target_is_directory=True)
    assert model_link.resolve() == snapshot
    value = {'repo_id': REPO, 'revision': REVISION, 'snapshot_path': str(snapshot),
        'model_path': str(model_link), 'model_safetensors_sha256': SHA256,
        'model_safetensors_bytes': SIZE, 'files': {path.name: path.stat().st_size for path in snapshot.iterdir()}}
    write_json(HERE / 'nonthinking_model_download.json', value)
    write_json(HERE / 'nonthinking_model_download_progress.json', {'state': 'complete', **value})
    # These are temporary segments of the newly downloaded file, not old model/data artifacts.
    for path in parts.glob('*.part'):
        path.unlink()
    print(json.dumps({'state': 'complete', **value}), flush=True)


if __name__ == '__main__':
    main()
