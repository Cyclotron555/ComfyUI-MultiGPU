"""Explicit, bounded RTX 3060 lower-power experiment with automatic restoration.

Dry-run by default. --run requires administrator authentication. This utility
never runs model code as root and never raises the cap above its original value.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SMI = '/usr/bin/nvidia-smi'


def query(primary):
    fields = ['uuid', 'name', 'memory.total', 'power.limit', 'power.min_limit', 'power.max_limit']
    line = subprocess.check_output([SMI, '--id=' + primary, '--query-gpu=' + ','.join(fields),
                                    '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    row = dict(zip(fields, (value.strip() for value in line.split(','))))
    if row.get('uuid') != primary or 'RTX 3060' not in row.get('name', '') or int(row['memory.total']) < 11000:
        raise ValueError('Expected the exact RTX 3060 12GB UUID; refusing a substitute GPU')
    return {'uuid': primary, 'name': row['name'], 'original_watts': float(row['power.limit']),
            'minimum_watts': float(row['power.min_limit']), 'maximum_watts': float(row['power.max_limit'])}


def set_limit(primary, watts):
    subprocess.run([SMI, '--id=' + primary, '--power-limit=' + format(watts, 'g')], check=True, timeout=10)


@contextmanager
def temporary_limit(primary, watts, *, reader=query, setter=set_limit):
    before = reader(primary)
    original = before['original_watts']
    if not before['minimum_watts'] <= watts < original <= before['maximum_watts']:
        raise ValueError('Only a lower cap within the reported GPU limits is permitted')
    try:
        setter(primary, watts)
        if reader(primary)['original_watts'] != watts:
            raise RuntimeError('Requested lower power cap was not applied')
        print(json.dumps({'phase': 'applied', 'primary': primary, 'watts': watts, 'restore_watts': original}), flush=True)
        yield before
    finally:
        # Attempt restoration even if setting/readback failed or the caller was interrupted.
        for attempt in range(3):
            try:
                setter(primary, original)
                if reader(primary)['original_watts'] != original:
                    raise RuntimeError('Original power limit not restored')
                print(json.dumps({'phase': 'restored', 'primary': primary, 'watts': original}), flush=True)
                break
            except (OSError, subprocess.SubprocessError, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', required=True, help='Exact RTX 3060 UUID')
    parser.add_argument('--watts', type=int, default=130)
    parser.add_argument('--seconds', type=int, default=900, help='Bounded duration, 1–7200 seconds; choose explicitly for long renders')
    parser.add_argument('--stop-file', type=Path, required=True, help='Existence ends the experiment and restores the cap')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.primary.startswith('GPU-') or not 1 <= args.seconds <= 7200:
        parser.error('Use an exact GPU UUID and duration between 1 and 7200 seconds')
    if args.stop_file.exists():
        parser.error('Stop file already exists; inspect prior experiment before proceeding')
    before = query(args.primary)
    if not before['minimum_watts'] <= args.watts < before['original_watts']:
        parser.error('Requested cap must be within the GPU limits and below its current cap')
    print(json.dumps({'plan': before, 'temporary_watts': args.watts, 'maximum_seconds': args.seconds,
                      'stop_file': str(args.stop_file), 'run': args.run}), flush=True)
    if not args.run:
        return
    if os.geteuid() != 0:
        parser.error('--run requires administrator authentication')
    def interrupted(signum, frame):
        raise InterruptedError('Power experiment interrupted; restoring original cap')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with temporary_limit(args.primary, args.watts):
        end = time.monotonic() + args.seconds
        while time.monotonic() < end and not args.stop_file.exists():
            time.sleep(0.5)


if __name__ == '__main__':
    main()
