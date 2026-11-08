"""User-process power trace and cleanup signal for an explicitly capped batch.

Reads the comparison controller's atomic state file. Never runs model code or
changes power itself; signals the separately authenticated helper to restore.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', required=True)
    parser.add_argument('--state-file', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=7200)
    args = parser.parse_args()
    if not args.primary.startswith('GPU-') or not 1 <= args.seconds <= 7200:
        parser.error('Use an exact GPU UUID and a bounded duration of 1–7200 seconds')
    if args.stop_file.exists():
        parser.error('Stop signal already exists; inspect previous run')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        with args.output.open('x') as log:
            while time.monotonic() - start < args.seconds:
                state = json.loads(args.state_file.read_text())
                if state['status'] in ('complete', 'failed') or args.stop_file.exists():
                    break
                result = subprocess.check_output([
                    '/usr/bin/nvidia-smi', '--id=' + args.primary,
                    '--query-gpu=uuid,power.limit,power.draw,temperature.gpu,clocks.sm',
                    '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
                identity, limit, draw, temperature, clock = [value.strip() for value in result.split(',')]
                if identity != args.primary:
                    raise RuntimeError('Power watcher GPU identity mismatch')
                log.write(json.dumps({'elapsed_seconds': round(time.monotonic()-start, 2),
                                      'active': state.get('active'), 'power_limit_watts': float(limit),
                                      'power_draw_watts': float(draw), 'temperature_c': int(temperature),
                                      'sm_mhz': int(clock)}) + '\n')
                log.flush()
                time.sleep(2)
    finally:
        args.stop_file.touch()  # restores the original cap even on watcher failure


if __name__ == '__main__':
    main()
