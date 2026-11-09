import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('wan_job_limits', TOOLS / 'wan_job_limits.py')
limits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(limits)


class JobLimitsTests(unittest.TestCase):
    def files(self, maximum=40, high=36, swap=0, oom='1'):
        relative = '/user.slice/codex-wan-safe-job.service'
        base = '/sys/fs/cgroup' + relative
        return {
            '/proc/self/cgroup': '0::' + relative + '\n',
            base + '/memory.max': str(maximum * limits.GIB) if isinstance(maximum, int) else maximum,
            base + '/memory.high': str(high * limits.GIB) if isinstance(high, int) else high,
            base + '/memory.swap.max': str(swap),
            base + '/memory.oom.group': oom,
        }

    def check(self, files):
        with patch.object(Path, 'read_text', lambda p, *a, **kw: files[str(p)]):
            return limits.require_isolated_job()

    def test_bounded_isolated_job_passes(self):
        result = self.check(self.files())
        self.assertEqual(result['memory_max_bytes'], 40 * limits.GIB)
        self.assertEqual(result['oom_group'], '1')

    def test_browser_scope_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'isolate'):
            self.check({'/proc/self/cgroup': '0::/app.slice/browser.scope\n'})

    def test_missing_or_excessive_memory_cap_is_rejected(self):
        for maximum in ['max', 41]:
            with self.subTest(maximum=maximum), self.assertRaises(RuntimeError):
                self.check(self.files(maximum=maximum))

    def test_swap_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check(self.files(swap=1))

    def test_missing_pressure_or_group_limits_are_rejected(self):
        for changes in [{'high': 'max'}, {'high': 40}, {'high': 0}, {'oom': '0'}]:
            with self.subTest(changes=changes), self.assertRaises(RuntimeError):
                self.check(self.files(**changes))

    def test_launcher_defaults_to_plan_without_starting_a_service(self):
        result = subprocess.run([sys.executable, str(TOOLS / 'wan_safe_job.py'), '--kind', 'render', '--', sys.executable, '-c', 'raise RuntimeError("must not run")'], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan['run'])
        self.assertEqual(plan['memory_max_bytes'], 40 * limits.GIB)
        self.assertEqual(plan['swap_max_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
