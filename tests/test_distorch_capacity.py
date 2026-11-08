"""Exercise actual allocation function without importing ComfyUI or initializing CUDA."""
import ast
from collections import defaultdict
import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_hardware_profile import policy, GIB

source = ast.parse((Path(__file__).resolve().parents[1] / 'distorch_2.py').read_text())
function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'calculate_safetensor_vvram_allocation')


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'MGPU_SAFE_OFFLOAD': '1', 'MGPU_PRIMARY_DEVICE': 'cuda:1'}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        totals = {'cuda:1': 12*GIB, 'cuda:0': 6*GIB, 'cpu': 64*GIB}
        free = {'cuda:1': 10*GIB, 'cuda:0': 3*GIB, 'cpu': 16*GIB}
        namespace = dict(os=os, math=math, logger=logging.getLogger('test'), memory_budget=policy.memory_budget,
                         torch=SimpleNamespace(device=lambda d: d),
                         mm=SimpleNamespace(get_total_memory=totals.__getitem__, get_free_memory=free.__getitem__),
                         get_device_list=lambda: list(totals))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<allocation>', 'exec'), namespace)
        self.allocate = namespace[function.name]
        self.model = SimpleNamespace(named_modules=lambda: [])

    def test_donor_capped_and_ram_fallback(self):
        result = dict(item.split(',') for item in self.allocate(self.model, 'cuda:1;5;cuda:0').split(';'))
        self.assertAlmostEqual(float(result['cuda:0']) * 6, 2, places=3)
        self.assertAlmostEqual(float(result['cpu']) * 64, 3, places=2)

    def test_exhausted_capacity_fails(self):
        with self.assertRaisesRegex(ValueError, 'Insufficient donor'):
            self.allocate(self.model, 'cuda:1;20;cuda:0')

    def test_self_donor_and_invalid_amount_fail(self):
        for spec in ['cuda:1;1;cuda:1', 'cuda:1;nan;cpu', 'cuda:1;-1;cpu', 'cuda:1;1;cuda:7']:
            with self.assertRaises(ValueError):
                self.allocate(self.model, spec)

class AssignmentGuardTests(unittest.TestCase):
    def test_expert_assignment_cannot_bypass_primary_reserve(self):
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze_safetensor_loading')
        module = SimpleNamespace(bias=None)
        namespace = dict(os=os, defaultdict=defaultdict, logger=logging.getLogger('test'),
                         memory_budget=policy.memory_budget, torch=SimpleNamespace(device=lambda d: d),
                         mm=SimpleNamespace(get_total_memory=lambda d: 12*GIB, get_free_memory=lambda d: 5*GIB),
                         get_device_list=lambda: ['cpu', 'cuda:1'], unpack_load_item=lambda item: item)
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<assignment>', 'exec'), namespace)
        model = SimpleNamespace(load_device='cuda:1', _load_list=lambda: [(4*GIB, 'block', module, [])])
        with patch.dict(os.environ, {'MGPU_SAFE_OFFLOAD': '1', 'MGPU_PRIMARY_DEVICE': 'cuda:1'}, clear=True):
            with self.assertRaisesRegex(ValueError, 'safe budget'):
                namespace[function.name](model, 'cuda:1,1.0')
