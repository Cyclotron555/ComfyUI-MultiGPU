import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('worker_thermal', Path(__file__).resolve().parents[1] / 'worker_thermal.py')
thermal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thermal)


class ThermalTests(unittest.TestCase):
    def pacer(self, temperatures, **kwargs):
        self.now = 0.0
        def sleep(seconds):
            self.now += seconds
        self.fence, self.drain = Mock(), Mock()
        return thermal.ThermalPacer('GPU-primary', fence=self.fence, drain=self.drain,
                                    temperature_reader=Mock(side_effect=temperatures),
                                    clock=lambda: self.now, sleep=sleep, **kwargs)

    def test_cold_check_does_not_drain_prefetch_streams(self):
        pacer = self.pacer([74])
        pacer.check()
        self.fence.assert_called_once()
        self.drain.assert_not_called()
        self.assertEqual(pacer.pauses, 0)

    def test_hot_check_drains_and_waits_for_resume_target(self):
        pacer = self.pacer([75, 76, 70, 67])
        pacer.check()
        self.drain.assert_called_once()
        self.assertEqual(pacer.cooling_seconds, 0.5)
        self.assertEqual(pacer.maximum_observed_c, 76)

    def test_explicit_77c_worker_pilot_keeps_cutoff_and_waits_to_72c(self):
        pacer = self.pacer([76, 77, 77, 73, 72], pause_at=77, resume_at=72)
        pacer.check()
        self.drain.assert_not_called()
        pacer.check()
        self.drain.assert_called_once()
        self.assertEqual(pacer.cooling_seconds, 0.5)
        self.assertEqual(pacer.cutoff, 82)

    def test_77c_worker_with_stricter_80c_stop(self):
        pacer = self.pacer([77, 78, 74, 72, 80], pause_at=77, resume_at=72, cutoff=80)
        pacer.check()
        self.assertEqual(pacer.cooling_seconds, 0.5)
        with self.assertRaisesRegex(RuntimeError, '80C cutoff'):
            pacer.check()

    def test_cutoff_before_or_after_drain_stops_worker(self):
        for values in [[82], [75, 82]]:
            with self.subTest(values=values), self.assertRaisesRegex(RuntimeError, 'cutoff'):
                self.pacer(values).check()

    def test_stuck_temperature_has_bounded_wait(self):
        pacer = self.pacer([76] * 20, timeout_seconds=1)
        with self.assertRaisesRegex(RuntimeError, 'timeout'):
            pacer.check()

    def test_bad_telemetry_does_not_silently_resume(self):
        for value in [float('nan'), -1]:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.pacer([value]).check()

    def test_invalid_settings_are_rejected(self):
        for settings in [{'pause_at': 83}, {'resume_at': 76}, {'poll_seconds': 0}, {'cutoff': float('inf')}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.pacer([50], **settings)

    def test_temperature_identity_mismatch_is_an_error(self):
        with patch.object(thermal.subprocess, 'check_output', return_value='GPU-other, 40'):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                thermal.read_temperature('GPU-primary')


if __name__ == '__main__':
    unittest.main()
