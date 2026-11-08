import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('policy', ROOT / 'hardware_profile.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
spec = importlib.util.spec_from_file_location('launcher', ROOT / 'safe_launch.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
GIB = 1024**3


class HardwarePolicyTests(unittest.TestCase):
    def test_core_primary_is_not_first_gpu(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(policy.primary_device(['cpu', 'cuda:0', 'cuda:1'], 'cuda:1'), 'cuda:1')

    def test_explicit_mapping_and_missing_device(self):
        with patch.dict(os.environ, {'MGPU_PRIMARY_DEVICE': 'cuda:1'}, clear=True):
            self.assertEqual(policy.primary_device(['cpu', 'cuda:1'], 'cpu'), 'cuda:1')
            with self.assertRaises(ValueError):
                policy.primary_device(['cpu'], 'cpu')

    def test_budget_uses_free_memory_and_reserves(self):
        with patch.dict(os.environ, {'MGPU_SAFE_OFFLOAD': '1', 'MGPU_PRIMARY_DEVICE': 'cuda:1'}, clear=True):
            self.assertEqual(policy.memory_budget('cuda:1', 12*GIB, 5*GIB), 3*GIB)
            self.assertEqual(policy.memory_budget('cuda:0', 6*GIB, 6*GIB), 5*GIB)
            self.assertEqual(policy.memory_budget('cpu', 64*GIB, 10*GIB), 2*GIB)
            self.assertEqual(policy.memory_budget('cpu', 64*GIB, GIB), 0)

    def test_invalid_reserve_fails(self):
        for value in ['nan', 'inf', '-1']:
            with patch.dict(os.environ, {'MGPU_SAFE_OFFLOAD': '1', 'MGPU_GPU_RESERVE_GIB': value}, clear=True):
                with self.assertRaises(ValueError):
                    policy.memory_budget('cuda:1', 12*GIB, 10*GIB)

    def test_opt_in_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(policy.memory_budget('cuda:1', 12*GIB, GIB), 12*GIB)

    def test_uuid_or_pci_never_index(self):
        rows = [{'uuid': 'GPU-primary', 'pci.bus_id': '00000000:06:00.0'},
                {'uuid': 'GPU-donor', 'pci.bus_id': '00000000:05:00.0'}]
        self.assertEqual(launcher.select_gpu(rows, 'GPU-primary'), rows[0])
        self.assertEqual(launcher.select_gpu(rows, '00000000:05:00.0'), rows[1])
        with self.assertRaises(ValueError):
            launcher.select_gpu(rows, '0')


if __name__ == '__main__':
    unittest.main()
