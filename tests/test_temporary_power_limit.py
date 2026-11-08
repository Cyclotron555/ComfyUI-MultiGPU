import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('temporary_power_limit', Path(__file__).resolve().parents[1] / 'benchmarks/temporary_power_limit.py')
power = importlib.util.module_from_spec(spec)
spec.loader.exec_module(power)


class PowerTests(unittest.TestCase):
    def scenario(self, initial=170):
        self.current = initial
        self.calls = []
        def read(primary):
            return {'original_watts': self.current, 'minimum_watts': 100, 'maximum_watts': 170}
        def set_limit(primary, watts):
            self.calls.append((primary, watts))
            self.current = watts
        return read, set_limit

    def test_restores_original_after_normal_completion(self):
        read, set_limit = self.scenario()
        with power.temporary_limit('GPU-primary', 130, reader=read, setter=set_limit):
            self.assertEqual(self.current, 130)
        self.assertEqual(self.current, 170)
        self.assertEqual(self.calls, [('GPU-primary', 130), ('GPU-primary', 170)])

    def test_restores_original_when_worker_or_user_interrupts(self):
        for error in [RuntimeError('worker failed'), KeyboardInterrupt()]:
            read, set_limit = self.scenario()
            with self.assertRaises(type(error)):
                with power.temporary_limit('GPU-primary', 130, reader=read, setter=set_limit):
                    raise error
            self.assertEqual(self.current, 170)

    def test_rejects_raising_or_out_of_range_cap_before_any_write(self):
        for watts in [99, 170, 180]:
            read, set_limit = self.scenario()
            with self.assertRaises(ValueError):
                with power.temporary_limit('GPU-primary', watts, reader=read, setter=set_limit):
                    pass
            self.assertEqual(self.calls, [])

    def test_identity_mismatch_does_not_select_another_card(self):
        line = 'GPU-other, NVIDIA GeForce RTX 3060, 12288, 170, 100, 170'
        with patch.object(power.subprocess, 'check_output', return_value=line):
            with self.assertRaisesRegex(ValueError, 'exact'):
                power.query('GPU-primary')


if __name__ == '__main__':
    unittest.main()
