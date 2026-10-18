"""CPU validation of completion accounting, committed IDs and safe GPU discovery."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'open_perfectblend_regen'))
sys.path.insert(0, str(HERE))
os.environ.update(USE_TORCH='0', USE_TF='0', USE_FLAX='0', CUDA_VISIBLE_DEVICES='')
import dispatch
from pool import write_json
import pilot_gpu1


class DispatchChecks(unittest.TestCase):
    def test_only_idle_compatible_devices_and_explicit_scope(self):
        xml = '<nvidia_smi_log>'
        for index, free, process in [(1, 81000, ''), (3, 40000, '<process_info><pid>123</pid><type>C</type></process_info>')]:
            xml += (f'<gpu><minor_number>{index}</minor_number><uuid>GPU-{index}</uuid><product_name>A100</product_name>'
                    f'<fb_memory_usage><free>{free} MiB</free><total>81920 MiB</total></fb_memory_usage>'
                    f'<processes>{process}</processes></gpu>')
        xml += '</nvidia_smi_log>'
        owner = {'pid': 123, 'uid': 1000, 'user': 'owner', 'ownership': 'self'}
        with patch.object(dispatch.subprocess, 'check_output', return_value=xml) as query:
            with patch.object(dispatch, 'process_owner', return_value=owner):
                devices = dispatch.gpu_inventory('1,3')
        self.assertIn('1,3', query.call_args.args[0])
        self.assertTrue(devices[0]['idle'])
        self.assertFalse(devices[1]['idle'])
        self.assertEqual(devices[1]['compute_pids'], [123])
        self.assertEqual(devices[1]['compute_processes'], [owner])

    def test_process_ownership_uses_os_uid_and_handles_unreadable_pid(self):
        current = dispatch.process_owner(os.getpid())
        self.assertEqual(current['ownership'], 'self')
        self.assertEqual(current['uid'], os.getuid())
        with patch.object(dispatch.Path, 'stat', return_value=SimpleNamespace(st_uid=os.getuid() + 1)):
            self.assertEqual(dispatch.process_owner(123)['ownership'], 'other')
        with patch.object(dispatch.Path, 'stat', side_effect=PermissionError('unreadable')):
            unknown = dispatch.process_owner(123)
        self.assertEqual(unknown['ownership'], 'unknown')
        self.assertIsNone(unknown['uid'])

    def test_live_client_adoption_does_not_cancel(self):
        client = pilot_gpu1.AttachedClient(123)
        with patch.object(pilot_gpu1, 'process_alive', side_effect=[True, False]):
            with patch.object(pilot_gpu1.os, 'killpg') as cancel:
                self.assertIsNone(client.poll())
                self.assertEqual(client.poll(), 0)
                cancel.assert_not_called()

    def test_live_client_handoff_never_repairs_or_scans_its_active_append(self):
        with patch.object(pilot_gpu1, 'repair_output_tail') as repair:
            with patch.object(pilot_gpu1.launcher, 'completed_count') as scan:
                result = pilot_gpu1.initial_job_counts({'output': 'actively-written.jsonl'},
                    {'existing_conversations': 17, 'existing_skipped_conversations': 2}, adopting=True)
                self.assertEqual(result, (17, 2))
                repair.assert_not_called()
                scan.assert_not_called()

    def test_occupied_service_port_uses_another_port(self):
        with patch.object(pilot_gpu1.socket, 'socket') as factory:
            factory.return_value.__enter__.return_value.bind.side_effect = [OSError('occupied'), None]
            self.assertEqual(pilot_gpu1.available_port(0), 18180)

    def test_training_inventory_keeps_absolute_data_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                root = Path('thinking')
                (root / 'train').mkdir(parents=True)
                (root / 'train/shard-000.jsonl').write_text('{}\n')
                (root / 'train/shard-001.jsonl').touch()
                (root / 'train/shard-000.skipped.jsonl').write_text('{}\n')
                (root / 'validation').mkdir()
                (root / 'validation/shard-000.jsonl').write_text('{}\n')
                dispatch.export_training_paths(root)
                result = json.loads((root / 'training_data_paths.json').read_text())
                self.assertEqual(result['train'], [str(Path(directory) / 'thinking/train/shard-000.jsonl')])
                self.assertEqual(result['validation'], [])
            finally:
                os.chdir(previous)

    def test_completion_requires_id_coverage_valid_rows_and_separate_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            root = directory / 'regenerated_ctx4096/nonthinking'
            shards = []
            for split, ids in [('train', ['a', 'b']), ('validation', ['v'])]:
                source = directory / f'prompts/{split}/shard-000.jsonl'
                source.parent.mkdir(parents=True)
                source.write_text(''.join(json.dumps({'id': key}) + '\n' for key in ids))
                shards.append({'split': split, 'shard': 0, 'rows': len(ids),
                               'path': str(source.relative_to(directory)),
                               'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
                if split == 'validation':
                    continue
                output = root / split / 'shard-000.jsonl'
                output.parent.mkdir(parents=True)
                key = ids[0]
                row = {'id': key + '_gen0', 'primary_id': key, 'input_ids': [1] * 5 + [2] * 8,
                       'loss_mask': [0] * 5 + [1] * 8, 'metadata': {'conversation_complete': True}}
                output.write_text(json.dumps(row) + '\n')
                write_json(output.with_suffix('.status.json'), {'state': 'complete',
                    'completed_conversations': 1, 'skipped_conversations': len(ids) - 1})
            write_json(directory / 'manifest.json', {'shards': shards})
            training = root / 'train/shard-000.jsonl'
            # Progress counters alone cannot prove all source IDs were processed.
            self.assertTrue(dispatch.pool_progress(directory)['all_processed'])
            self.assertEqual(dispatch.pool_progress(directory)['total_shards'], 1)
            incomplete = dispatch.verify_outputs(directory)
            self.assertFalse(incomplete['all_processed'])
            self.assertEqual(incomplete['splits']['train']['remaining_conversations'], 1)
            training.with_suffix('.skipped.jsonl').write_text(json.dumps(
                {'primary_id': 'b', 'terminal_skip': True, 'reason': 'prompt fills context'}) + '\n')
            verified = dispatch.verify_outputs(directory)
            self.assertTrue(verified['all_processed'])
            self.assertFalse(verified['all_generated_without_skips'])
            self.assertEqual(verified['state'], 'complete_with_skips')
            self.assertEqual(verified['active_splits'], ['train'])
            self.assertNotIn('validation', verified['splits'])
            self.assertEqual(verified['splits']['train']['generated_conversations'], 1)
            self.assertEqual(verified['splits']['train']['skipped_conversations'], 1)
            original = training.read_text()
            training.write_text(original * 2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                dispatch.verify_outputs(directory)
            row = json.loads(original)
            row['loss_mask'] = [0] * len(row['input_ids'])
            training.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'Invalid 4096'):
                dispatch.verify_outputs(directory)
            row['input_ids'] = [1] * 4090 + [2] * 8
            row['loss_mask'] = [0] * 4090 + [1] * 8
            training.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'Invalid 4096'):
                dispatch.verify_outputs(directory)


if __name__ == '__main__':
    unittest.main(verbosity=2)
