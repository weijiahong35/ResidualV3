"""CPU checks for the authorized Beijing GPU windows and release boundary."""
import datetime as dt
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'open_perfectblend_regen'))
import scheduler


class CapacityChecks(unittest.TestCase):
    def local(self, value):
        return dt.datetime.fromisoformat(value)

    def test_beijing_midnight_and_0800_boundaries(self):
        for time, permitted, effective in [
            ('2026-10-04T23:59:59+08:00', 3, 3),
            ('2026-10-05T00:00:00+08:00', 6, 6),
            ('2026-10-05T05:59:00+08:00', 6, 6),
            ('2026-10-05T06:00:00+08:00', 6, 6),
            ('2026-10-05T07:58:59+08:00', 6, 6),
            ('2026-10-05T07:59:00+08:00', 6, 3),
            ('2026-10-05T07:59:59+08:00', 6, 3),
            ('2026-10-05T08:00:00+08:00', 3, 3)]:
            with self.subTest(time=time):
                policy = scheduler.capacity_at(self.local(time))
                self.assertEqual(policy['permitted_limit'], permitted)
                self.assertEqual(policy['effective_limit'], effective)
                self.assertEqual(policy['night_window'], '00:00-08:00')
                self.assertEqual(policy['draining_before_daytime'], permitted != effective)

    def test_utc_input_is_converted_to_beijing(self):
        policy = scheduler.capacity_at(self.local('2026-10-04T16:00:00+00:00'))
        self.assertEqual(policy['effective_limit'], 6)
        self.assertEqual(policy['local_time'], '2026-10-05T00:00:00+08:00')
        self.assertEqual(policy['next_change_at_local'], '2026-10-05T07:59:00+08:00')

    def test_next_event_crosses_year_and_sleep_stops_at_boundary(self):
        policy = scheduler.capacity_at(self.local('2026-12-31T23:59:59+08:00'))
        self.assertEqual(policy['next_change_at_local'], '2027-01-01T00:00:00+08:00')
        self.assertEqual(scheduler.poll_delay(20, policy), 1)
        with self.assertRaisesRegex(ValueError, 'timezone-aware'):
            scheduler.capacity_at(dt.datetime(2026, 10, 4))

    def devices(self, workers, idle):
        return [{'index': index, 'idle': index in range(workers, workers + idle),
                 'compatible': True} for index in range(8)]

    def test_daytime_uses_last_idle_gpu_and_retains_three_when_full(self):
        policy = scheduler.capacity_at(self.local('2026-10-05T18:00:00+08:00'))
        for workers, idle in [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (4, 0)]:
            with self.subTest(workers=workers, idle=idle):
                result = scheduler.allocation_at(policy, self.devices(workers, idle), range(workers))
                self.assertEqual(result['target_workers'], 3)
                self.assertEqual(result['minimum_retained_workers'], 2)
                self.assertEqual(result['idle_gpus'], idle)
                self.assertEqual(result['reserved_idle_gpus_above_minimum'], 0)
                self.assertFalse(result['release_worker_when_all_gpus_busy'])
        # A single idle card can fill the third slot without an idle-card reserve.
        result = scheduler.allocation_at(policy, self.devices(2, 1), range(2))
        self.assertEqual(result['idle_gpu_indices'], [2])
        self.assertGreater(result['target_workers'], 2)

    def test_starting_workers_are_not_counted_twice_as_idle_gpus(self):
        policy = scheduler.capacity_at(self.local('2026-10-04T18:00:00+08:00'))
        devices = self.devices(2, 1)
        for device in devices[:2]:
            device['idle'] = True  # Worker exists but its model has not acquired memory yet.
        result = scheduler.allocation_at(policy, devices, range(2))
        self.assertEqual(result['idle_gpu_indices'], [2])
        self.assertEqual(result['target_workers'], 3)

    def test_night_cap_and_morning_transition_enforce_only_time_limits(self):
        night = scheduler.capacity_at(self.local('2026-10-05T07:58:59+08:00'))
        self.assertEqual(scheduler.allocation_at(night, self.devices(6, 0), range(6))['target_workers'], 6)
        draining = scheduler.capacity_at(self.local('2026-10-05T07:59:00+08:00'))
        self.assertEqual(scheduler.allocation_at(draining, self.devices(6, 0), range(6))['target_workers'], 3)
        daytime = scheduler.capacity_at(self.local('2026-10-05T08:00:00+08:00'))
        self.assertEqual(scheduler.allocation_at(daytime, self.devices(3, 1), range(3))['target_workers'], 3)
        fixed_two = scheduler.capacity_at(self.local('2026-10-05T08:00:00+08:00'), day_limit=2)
        self.assertEqual(scheduler.allocation_at(fixed_two, self.devices(2, 6), range(2))['target_workers'], 2)
        previous_night_limit = scheduler.capacity_at(self.local('2026-10-05T01:00:00+08:00'), night_limit=4)
        self.assertEqual(previous_night_limit['effective_limit'], 4)

    def test_full_occupancy_does_not_reduce_three_for_any_process_owner(self):
        policy = scheduler.capacity_at(self.local('2026-10-05T20:00:00+08:00'))
        for owner in ['self', 'other', 'unknown']:
            with self.subTest(owner=owner):
                devices = self.devices(3, 0)
                for device in devices:
                    device['compute_processes'] = [{'pid': 100 + device['index'], 'ownership': owner}]
                result = scheduler.allocation_at(policy, devices, range(3))
                self.assertEqual(result['target_workers'], 3)
                self.assertFalse(result['release_worker_when_all_gpus_busy'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
