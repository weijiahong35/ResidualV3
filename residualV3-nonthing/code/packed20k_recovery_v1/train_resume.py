"""Resume the immutable v3 trainer with a versioned HTTP transport fix."""
import importlib.util
import os
from pathlib import Path
import sys

ORIGINAL = Path(__file__).resolve().parent.parent / 'packed20k_split_v3'
sys.path.insert(0, str(ORIGINAL / 'runtime_bundle'))
sys.path.insert(0, str(ORIGINAL / 'runtime_bundle/src'))
sys.path.insert(0, str(ORIGINAL / 'runtime_bundle/hs_connectors/src'))
from transport import install

install(Path(os.environ['PACKED_RECOVERY_ATTEMPT']) / 'hidden_connection_retries.jsonl')
spec = importlib.util.spec_from_file_location('immutable_packed20k_train', ORIGINAL / 'train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main()
