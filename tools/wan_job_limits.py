"""Require our own isolated systemd service before starting another heavy job."""
from pathlib import Path

GIB = 1024**3
UNIT = 'codex-wan-safe-job.service'


def require_isolated_job(max_gib=40):
    relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    if Path(relative).name != UNIT:
        raise RuntimeError('Job blocked: use tools/wan_safe_job.py to isolate it from browser applications')
    group = Path('/sys/fs/cgroup') / relative.lstrip('/')
    maximum = (group / 'memory.max').read_text().strip()
    swap = (group / 'memory.swap.max').read_text().strip()
    high = (group / 'memory.high').read_text().strip()
    oom_group = (group / 'memory.oom.group').read_text().strip()
    if maximum == 'max' or int(maximum) > max_gib * GIB or swap != '0':
        raise RuntimeError('Job blocked: missing required memory/swap limits')
    if high == 'max' or not 0 < int(high) < int(maximum) or oom_group != '1':
        raise RuntimeError('Job blocked: missing required memory pressure/OOM group limits')
    return {'cgroup': relative, 'memory_max_bytes': int(maximum),
            'memory_high_bytes': int(high),
            'memory_swap_max_bytes': int(swap),
            'oom_group': oom_group}
