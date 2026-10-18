import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'open_perfectblend_regen'))
from speculation_policy import with_tokens, selected_tokens, execution_record, choose_result


class SpeculationPolicyChecks(unittest.TestCase):
    def setUp(self):
        self.spec = {'method':'dflash','model':'/pinned/nonthinking',
                     'num_speculative_tokens':15,'draft_sample_method':'greedy'}
        self.command = ['python','vllm','--max-model-len','4096',
                        '--speculative-config',json.dumps(self.spec)]

    def test_only_draft_length_changes_and_native_limit_is_enforced(self):
        cmd, actual = with_tokens(self.command, 5)
        self.assertEqual(cmd[:-1], self.command[:-1])
        self.assertEqual(actual, {**self.spec, 'num_speculative_tokens':5})
        for invalid in [0,16,True,3.5]:
            with self.assertRaises(ValueError):
                with_tokens(self.command, invalid)
        with self.assertRaisesRegex(ValueError,'pinned draft'):
            bad = {**actual, 'model':'/different'}
            execution_record({'command':self.command[:-1]+[json.dumps(bad)]},
                             {'speculative_config':self.spec}, '/policy')

    def test_per_gpu_trial_precedes_global_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile = SimpleNamespace(output_root=root,worker_dir=lambda i:root/f'gpu{i}')
            (root/'gpu1').mkdir()
            self.assertEqual(selected_tokens(profile,1,15),15)
            (root/'speculation_policy.json').write_text(json.dumps({'num_speculative_tokens':7}))
            self.assertEqual(selected_tokens(profile,1,15),7)
            (root/'gpu1/speculation.json').write_text(json.dumps({'num_speculative_tokens':3}))
            self.assertEqual(selected_tokens(profile,1,15),3)
            self.assertEqual(selected_tokens(profile,2,15),7)

    def test_selection_uses_throughput_and_refuses_failed_or_incomplete_trials(self):
        def trial(k,tps,errors=0,processed=128):
            return {'num_speculative_tokens':k,'output_tokens_per_second':tps,
                    'errors':errors,'processed':processed,'expected':128}
        rows=[trial(15,1700),trial(7,2000),trial(5,2200),trial(3,2500,errors=1)]
        best,gain=choose_result(rows)
        self.assertEqual(best['num_speculative_tokens'],5)
        self.assertGreater(gain,.25)
        best,_=choose_result([trial(15,1700),trial(3,1750)])
        self.assertEqual(best['num_speculative_tokens'],15)


if __name__ == '__main__':
    unittest.main()
