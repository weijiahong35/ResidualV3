"""Regression checks for crashes and interrupted live concurrency tuning."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'open_perfectblend_regen'))
from concurrency_policy import record_server_failure, tuning_candidates


class ConcurrencyRecoveryChecks(unittest.TestCase):
    def test_crash_at_128_does_not_repeat_after_json_restart(self):
        measured = {'results': [{'concurrency': n, 'score_tokens_per_second': rate}
                               for n, rate in [(40, 1766), (64, 1722), (80, 1576), (96, 1785)]]}
        self.assertEqual(tuning_candidates(measured), [128])
        recovered = json.loads(json.dumps(record_server_failure(measured, 128, 'CUDA illegal memory access')))
        self.assertEqual(recovered['selected_concurrency'], 40)
        self.assertEqual(tuning_candidates(recovered), [])
        self.assertEqual(recovered['failures'][0]['concurrency'], 128)

    def test_interrupted_measurements_resume_untested_batch_sizes(self):
        self.assertEqual(tuning_candidates({'results': [{'concurrency': 40}, {'concurrency': 80}]}),
                         [64, 96, 128])

    def test_failed_selected_size_backs_off(self):
        recovered = record_server_failure({'results': [], 'selected_concurrency': 40}, 40, 'server died')
        self.assertEqual(recovered['selected_concurrency'], 20)
        self.assertEqual(tuning_candidates(recovered), [])

    def test_selected_partial_results_remain_final(self):
        self.assertEqual(tuning_candidates({'selected_concurrency': 40, 'results': [{'concurrency': 40}]}), [])


if __name__ == '__main__':
    unittest.main()
