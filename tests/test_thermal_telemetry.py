import ctypes
import importlib.util
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


telemetry = load('thermal_telemetry', ROOT / 'thermal_telemetry.py')
power = load('temporary_power_limit', ROOT / 'benchmarks/temporary_power_limit.py')


class FakeNvml:
    def __init__(self, identity='GPU-primary'):
        self.identity, self.temperature = identity, 60
        self.policies, self.targets = [0, 0], [35, 35]
        self.calls, self.fail_fan = [], None
        names = ['nvmlInit_v2', 'nvmlShutdown', 'nvmlDeviceGetHandleByUUID',
                 'nvmlDeviceGetUUID', 'nvmlDeviceGetTemperature', 'nvmlDeviceGetNumFans',
                 'nvmlDeviceGetMinMaxFanSpeed', 'nvmlDeviceGetFanControlPolicy_v2',
                 'nvmlDeviceGetTargetFanSpeed', 'nvmlDeviceSetFanSpeed_v2',
                 'nvmlDeviceSetDefaultFanSpeed_v2']
        for name in names:
            setattr(self, name, Mock(side_effect=lambda *args, name=name: self.call(name, *args)))

    def call(self, name, *args):
        self.calls.append((name, args))
        if name == 'nvmlDeviceGetHandleByUUID': args[-1]._obj.value = 123
        if name == 'nvmlDeviceGetUUID': args[1].value = self.identity.encode()
        if name == 'nvmlDeviceGetTemperature': args[-1]._obj.value = self.temperature
        if name == 'nvmlDeviceGetNumFans': args[-1]._obj.value = 2
        if name == 'nvmlDeviceGetMinMaxFanSpeed':
            args[-2]._obj.value, args[-1]._obj.value = 30, 100
        if name == 'nvmlDeviceGetFanControlPolicy_v2': args[-1]._obj.value = self.policies[args[1]]
        if name == 'nvmlDeviceGetTargetFanSpeed': args[-1]._obj.value = self.targets[args[1]]
        if name == 'nvmlDeviceSetFanSpeed_v2':
            self.targets[args[1]], self.policies[args[1]] = args[2], 1
            if args[1] == self.fail_fan: return 999
        if name == 'nvmlDeviceSetDefaultFanSpeed_v2': self.policies[args[1]] = 0
        return 0


class TelemetryTests(unittest.TestCase):
    def test_live_reader_verifies_uuid_and_balances_shutdown(self):
        lib = FakeNvml()
        with telemetry.NvmlTemperatureReader('GPU-primary', library_factory=lambda _: lib) as reader:
            self.assertEqual(reader(), 60)
            lib.temperature = 75
            self.assertEqual(reader(), 75)  # no stale temperature cache
        lib.nvmlShutdown.assert_called_once()
        with self.assertRaises(RuntimeError): reader()
        self.assertFalse(any('Set' in name for name, _ in lib.calls))

    def test_wrong_uuid_cannot_silently_select_a_different_gpu(self):
        lib = FakeNvml('GPU-other')
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            telemetry.NvmlTemperatureReader('GPU-primary', library_factory=lambda _: lib)
        lib.nvmlShutdown.assert_called_once()

    def test_query_error_and_fork_are_fail_closed(self):
        lib = FakeNvml()
        reader = telemetry.NvmlTemperatureReader('GPU-primary', library_factory=lambda _: lib)
        lib.nvmlDeviceGetTemperature.side_effect = None
        lib.nvmlDeviceGetTemperature.return_value = 15
        with self.assertRaisesRegex(RuntimeError, 'status 15'): reader()
        with patch.object(telemetry.os, 'getpid', return_value=-1):
            with self.assertRaisesRegex(RuntimeError, 'another process'): reader()
        reader.close()

    def test_import_has_no_driver_initialization(self):
        with patch.object(ctypes, 'CDLL', side_effect=AssertionError('unexpected GPU access')):
            load('uninitialized_telemetry', ROOT / 'thermal_telemetry.py')


class FanTests(unittest.TestCase):
    def session(self, lib):
        return power.TemporaryFans('GPU-primary', 75, library_factory=lambda _: lib)

    def test_read_only_plan_never_writes_and_releases_nvml(self):
        lib = FakeNvml(); session = self.session(lib); session.close()
        self.assertFalse(any('Set' in name for name, _ in lib.calls))
        lib.nvmlShutdown.assert_called_once()

    def test_both_fans_restored_to_automatic(self):
        lib = FakeNvml(); session = self.session(lib)
        session.apply(); session.check()
        self.assertEqual(lib.policies, [1, 1])
        self.assertEqual(lib.targets, [75, 75])
        session.close()
        self.assertEqual(lib.policies, [0, 0])

    def test_partial_set_failure_still_restores_both_fans(self):
        lib = FakeNvml(); lib.fail_fan = 1
        session = self.session(lib)
        try:
            with self.assertRaisesRegex(RuntimeError, '999'): session.apply()
        finally: session.close()
        self.assertEqual(lib.policies, [0, 0])

    def test_existing_manual_policy_is_not_replaced(self):
        lib = FakeNvml(); lib.policies[1] = 1
        with self.assertRaisesRegex(RuntimeError, 'manual'): self.session(lib)
        self.assertFalse(any('Set' in name for name, _ in lib.calls))

    def test_never_lowers_existing_automatic_target(self):
        lib = FakeNvml(); session = self.session(lib); lib.targets[0] = 80
        try:
            with self.assertRaisesRegex(RuntimeError, 'exceeds'): session.apply()
        finally: session.close()
        self.assertEqual(lib.policies, [0, 0])

    def test_overtemperature_and_changed_policy_stop_experiment(self):
        lib = FakeNvml(); session = self.session(lib); session.apply()
        try:
            lib.temperature = 80
            with self.assertRaisesRegex(RuntimeError, '80C'): session.check()
            lib.temperature = 65; lib.policies[0] = 0
            with self.assertRaisesRegex(RuntimeError, 'changed'): session.check()
        finally: session.close()

    def test_main_interrupt_restores_fans_power_and_notifies_controller(self):
        lib = FakeNvml(); fans = self.session(lib)
        watts = [170]
        def query(primary):
            return dict(uuid=primary, name='RTX 3060', original_watts=watts[0],
                        minimum_watts=100, maximum_watts=170)
        def setter(primary, value): watts[0] = value
        original_context = power.temporary_limit
        def context(primary, value):
            return original_context(primary, value, reader=query, setter=setter)
        with tempfile.TemporaryDirectory() as directory:
            stop, abort = Path(directory) / 'done', Path(directory) / 'abort'
            argv = ['helper', '--primary', 'GPU-primary', '--watts', '130', '--seconds', '10',
                    '--stop-file', str(stop), '--abort-file', str(abort), '--fan-percent', '75', '--run']
            with patch.object(sys, 'argv', argv), \
                 patch.object(power, 'query', side_effect=query), \
                 patch.object(power, 'temporary_limit', side_effect=context), \
                 patch.object(power, 'TemporaryFans', return_value=fans), \
                 patch.object(power.os, 'geteuid', return_value=0), \
                 patch.object(power.time, 'sleep', side_effect=KeyboardInterrupt), \
                 patch.object(power.signal, 'signal'):
                with self.assertRaises(KeyboardInterrupt): power.main()
            self.assertEqual(watts[0], 170)
            self.assertEqual(lib.policies, [0, 0])
            self.assertTrue(abort.exists())


if __name__ == '__main__':
    unittest.main()
