"""Exercise memory cleanup with and without a running PromptServer, without CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

source = ast.parse((Path(__file__).resolve().parents[1] / 'model_management_mgpu.py').read_text())
names = {'trigger_executor_cache_reset', 'check_cpu_memory_threshold', 'force_full_system_cleanup'}
functions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names]


class HeadlessCleanupTests(unittest.TestCase):
    def setUp(self):
        self.prompt_server = SimpleNamespace()
        logger = Mock(spec=['debug', 'info', 'warning', 'mgpu_mm_log'])
        self.namespace = dict(
            server=SimpleNamespace(PromptServer=self.prompt_server), logger=logger,
            psutil=SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=90, used=1024)),
            mm=SimpleNamespace(current_loaded_models=[]), multigpu_memory_log=Mock(),
            CPU_MEMORY_THRESHOLD_PERCENT=85, CPU_RESET_HYSTERESIS_PERCENT=5,
            _last_cpu_usage_at_reset=0)
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<cleanup>', 'exec'), self.namespace)

    def test_absent_server_instance_does_not_crash_cleanup(self):
        self.namespace['check_cpu_memory_threshold']()
        self.namespace['trigger_executor_cache_reset']()
        self.assertIn('Cleanup requested', self.namespace['force_full_system_cleanup']())

    def test_explicitly_uninitialized_server_does_not_crash(self):
        self.prompt_server.instance = None
        self.namespace['check_cpu_memory_threshold']()
        self.namespace['trigger_executor_cache_reset']()
        self.namespace['force_full_system_cleanup']()

    def test_active_execution_is_not_reset(self):
        queue = SimpleNamespace(currently_running=True, set_flag=Mock())
        self.prompt_server.instance = SimpleNamespace(prompt_queue=queue)
        self.namespace['check_cpu_memory_threshold']()
        self.namespace['trigger_executor_cache_reset']()
        self.namespace['force_full_system_cleanup'](force=False)
        queue.set_flag.assert_not_called()

    def test_idle_high_memory_resets_once_until_usage_rises(self):
        queue = SimpleNamespace(currently_running=False, set_flag=Mock())
        self.prompt_server.instance = SimpleNamespace(prompt_queue=queue)
        self.namespace['check_cpu_memory_threshold']()
        self.namespace['check_cpu_memory_threshold']()
        queue.set_flag.assert_called_once_with('free_memory', True)
