"""Verify append-only sampling, immutable source provenance, and legacy isolation."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'open_perfectblend_regen'))
from extend_pool import additional_shards
from pool_manifest import load_pool_manifest
from hot_pool_extension import identity


class PoolExtensionChecks(unittest.TestCase):
    def test_exiting_process_with_empty_cmdline_is_not_signalled_as_worker(self):
        stat_file, command_file = MagicMock(), MagicMock()
        stat_file.read_text.return_value = '123 (python) R ' + '0 ' * 18 + '12345'
        command_file.read_bytes.return_value = b''
        with patch('hot_pool_extension.Path') as path:
            path.return_value.__truediv__.side_effect = lambda name: {'stat': stat_file, 'cmdline': command_file}[name]
            self.assertIsNone(identity(123, ['pilot_gpu1.py']))

    def fixture(self, directory):
        def shard(split, n):
            return {'split': split, 'shard': n, 'rows': 2, 'sha256': f'hash-{split}-{n}',
                    'path': f'prompts/{split}/shard-{n:03d}.jsonl'}
        base = {'dataset': 'test-source', 'parameters': {'train_size': 4, 'validation_size': 2,
                'seed': 42, 'revision': 'a' * 40, 'shard_size': 2},
                'shards': [shard('train', 0), shard('train', 1), shard('validation', 0)],
                'source_counts_selected': {'train': {'math': 4}, 'validation': {'math': 2}}}
        raw = (json.dumps(base) + '\n').encode()
        digest = hashlib.sha256(raw).hexdigest()
        (directory / 'base.json').write_bytes(raw)
        extended = copy.deepcopy(base)
        extended['parameters']['train_size'] = 8
        extended['shards'].extend([shard('train', 2), shard('train', 3)])
        extended['source_counts_selected']['train']['math'] = 8
        extended['extension'] = {'base_manifest': 'base.json', 'base_manifest_sha256': digest}
        (directory / 'manifest.json').write_text(json.dumps(extended) + '\n')
        return base, extended, digest

    def test_full_eligible_pool_appends_every_unused_row_once(self):
        records = [(i, f'source-{i % 8}', b'group') for i in range(80)]
        original = set(range(24))
        counts = Counter(source for i, source, _ in records if i in original)
        chunks, added, final = additional_shards(records, original, counts, 80, 16, 42)
        selected = [i for chunk in chunks for i in chunk]
        self.assertEqual(set(selected), set(range(24, 80)))
        self.assertEqual(len(selected), 56)
        self.assertEqual(final, {f'source-{i}': 10 for i in range(8)})
        self.assertEqual(added, {f'source-{i}': 7 for i in range(8)})
        self.assertEqual([len(c) for c in chunks], [16, 16, 16, 8])
        for chunk in chunks:
            self.assertEqual({records[i][1] for i in chunk}, set(final))
        self.assertEqual(chunks, additional_shards(records, original, counts, 80, 16, 42)[0])

    def test_extended_plan_keeps_recipe_anchor_and_legacy_pool_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            base, extended, digest = self.fixture(directory)
            active, anchor, current = load_pool_manifest(directory)
            self.assertEqual(active['parameters']['train_size'], 8)
            self.assertEqual(anchor, digest)
            self.assertNotEqual(anchor, current)
            legacy, old_anchor, _ = load_pool_manifest(directory, prefer_base=True)
            self.assertEqual(legacy, base)
            self.assertEqual(old_anchor, digest)

    def test_changed_original_shard_or_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, extended, _ = self.fixture(directory)
            changed = copy.deepcopy(extended)
            changed['shards'][0]['sha256'] = 'different'
            (directory / 'manifest.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'original prompt shard'):
                load_pool_manifest(directory)
            changed = copy.deepcopy(extended)
            changed['parameters']['seed'] = 43
            (directory / 'manifest.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'recipe'):
                load_pool_manifest(directory)

    def test_altered_anchor_or_noncontiguous_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _, extended, _ = self.fixture(directory)
            changed = copy.deepcopy(extended)
            changed['shards'][-1]['shard'] = 4
            (directory / 'manifest.json').write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'contiguous'):
                load_pool_manifest(directory)
            (directory / 'manifest.json').write_text(json.dumps(extended))
            (directory / 'base.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                load_pool_manifest(directory)


if __name__ == '__main__':
    unittest.main()
