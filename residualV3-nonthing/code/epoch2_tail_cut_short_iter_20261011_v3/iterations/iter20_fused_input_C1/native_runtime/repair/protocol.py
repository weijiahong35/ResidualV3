import json
from pathlib import Path
from runtime.utils import sha

def read(path):
    p=json.loads(Path(path).read_text())
    for name,value in p['source_files'].items():
        assert sha(name)==value,name
    for name,value in p['asset_sha256'].items():
        assert sha(name)==value,name
    return p
