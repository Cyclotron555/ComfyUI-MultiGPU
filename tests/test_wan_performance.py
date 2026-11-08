"""CPU-only boundary tests; no CUDA initialization or model downloads."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wan_performance', ROOT / 'wan_performance.py')
performance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(performance)


class FakeWan:
    pass


FakeWan.__module__ = 'comfy.ldm.wan.model'


class Model:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model=FakeWan())
        self.load_device = 'cuda:1'
        self.attention = None

    def clone(self):
        result = Model()
        result.model = self.model
        result.load_device = self.load_device
        return result

    def set_model_optimized_attention(self, function):
        self.attention = function


class PerformanceTests(unittest.TestCase):
    def modules(self, available=True, supported=True, cache=True):
        functions = {'pytorch': 'native-pytorch', 'comfy_kitchen_int8': 'native-int8' if available else None}
        def execute(model, **kwargs):
            model.cache_options = kwargs
            return [model]
        modules = {
            'comfy.ldm.modules.attention': SimpleNamespace(get_attention_function=lambda name, default=None: functions.get(name, default)),
            'comfy_kitchen': SimpleNamespace(int8_attention_is_available=lambda device: supported),
            'comfy_extras.nodes_easycache': SimpleNamespace(EasyCacheNode=SimpleNamespace(execute=execute)) if cache else SimpleNamespace(),
        }
        return patch.object(performance, 'import_module', side_effect=lambda name: modules[name])

    def test_default_leaves_original_and_device_mapping_untouched(self):
        original = Model()
        with self.modules():
            result, report = performance.configure_wan(original)
        self.assertIsNone(original.attention)
        self.assertIs(result.model, original.model)
        self.assertEqual(result.load_device, 'cuda:1')
        self.assertEqual(report['compute_device'], 'cuda:1')
        self.assertEqual(report['cache_threshold'], 0)

    def test_requested_native_int8_and_cache_are_scoped_to_clone(self):
        original = Model()
        with self.modules():
            result, report = performance.configure_wan(original, attention='kitchen_int8', cache_threshold=0.05)
        self.assertEqual(result.attention, 'native-int8')
        self.assertEqual(result.cache_options['reuse_threshold'], 0.05)
        self.assertEqual(result.cache_options['start_percent'], 0.15)
        self.assertIsNone(original.attention)
        self.assertEqual(report['attention'], 'kitchen_int8')

    def test_unavailable_backend_and_unsupported_compute_device_fall_back(self):
        for available, supported in [(False, True), (True, False)]:
            with self.modules(available=available, supported=supported):
                result, report = performance.configure_wan(Model(), attention='kitchen_int8')
            self.assertEqual(result.attention, 'native-pytorch')
            self.assertEqual(len(report['fallback_reasons']), 1)

    def test_missing_optional_cache_is_reported(self):
        with self.modules(cache=False):
            _, report = performance.configure_wan(Model(), cache_threshold=0.05)
        self.assertEqual(report['cache_threshold'], 0)
        self.assertIn('cache disabled', report['fallback_reasons'][0])

    def test_invalid_inputs_fail_before_optional_imports(self):
        cases = [{'attention': 'unknown'}, {'cache_threshold': float('nan')},
                 {'cache_threshold': float('inf')}, {'cache_threshold': -1},
                 {'cache_start': 0.9, 'cache_end': 0.1}]
        with patch.object(performance, 'import_module') as imports:
            for case in cases:
                with self.assertRaises(ValueError):
                    performance.configure_wan(Model(), **case)
            imports.assert_not_called()

    def test_non_wan_model_is_rejected(self):
        model = Model()
        model.model.diffusion_model = object()
        with self.assertRaisesRegex(ValueError, 'Wan'):
            performance.configure_wan(model)

    def test_launch_precision_is_explicit_and_baseline_unchanged(self):
        self.assertEqual(performance.launch_flags(False), [])
        self.assertIn('--bf16-unet', performance.launch_flags(True))
        self.assertNotIn('--use-sage-attention', performance.launch_flags(True))


if __name__ == '__main__':
    unittest.main()
