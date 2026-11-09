"""Run one isolated, bounded job; printing a plan is the default."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path.cwd()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kind', choices=['check', 'download', 'prepare', 'render'], required=True)
parser.add_argument('--run', action='store_true')
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
limits = {'check': (128*1024**2, 96*1024**2), 'download': (8*1024**3, 6*1024**3),
          'prepare': (24*1024**3, 20*1024**3), 'render': (40*1024**3, 36*1024**3)}
maximum, high = limits[args.kind]
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if args.kind == 'check':
    code = "import sys,json;sys.path.insert(0,"+repr(str(Path(__file__).parent))+ ");from wan_job_limits import require_isolated_job;print(json.dumps(require_isolated_job()),flush=True)"
    command = [sys.executable, '-c', code]
elif not command:
    parser.error('Supply a command after --')
launch = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--service-type=exec',
          '--unit=codex-wan-safe-job', '--working-directory='+str(ROOT),
          '--property=MemoryMax='+str(maximum), '--property=MemoryHigh='+str(high),
          '--property=MemorySwapMax=0', '--property=OOMPolicy=kill',
          '--property=KillMode=control-group', '--property=RuntimeMaxSec=1800']
for name in ('CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','MGPU_PRIMARY_DEVICE','MGPU_SAFE_OFFLOAD',
             'HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','DO_NOT_TRACK'):
    if name in os.environ:
        launch.append('--setenv='+name+'='+os.environ[name])
launch += ['--setenv=HF_HUB_DISABLE_TELEMETRY=1', '--setenv=DO_NOT_TRACK=1',
           '--setenv=HF_XET_NUM_CONCURRENT_RANGE_GETS=2',
           '--setenv=HF_XET_CHUNK_CACHE_SIZE_BYTES=0', '--'] + command
print(json.dumps({'kind':args.kind,'run':args.run,'memory_max_bytes':maximum,
                  'memory_high_bytes':high,'swap_max_bytes':0,'unit':'codex-wan-safe-job.service'}),flush=True)
if args.run:
    # A fixed unit name refuses a second concurrent job; the service owns the
    # workload separately from the browser, even if this launcher disappears.
    result = subprocess.run(launch)
    raise SystemExit(result.returncode)
