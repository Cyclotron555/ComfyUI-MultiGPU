"""Explicit, bounded RTX 3060 lower-power experiment with automatic restoration.

Dry-run by default. --run requires administrator authentication. This utility
never runs model code as root and never raises the cap above its original value.
"""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time

SMI = '/usr/bin/nvidia-smi'


class TemporaryFans:
    """Opt-in NVML fan control; refuse to replace an existing manual policy."""
    def __init__(self, primary, percent, *, library_factory=ctypes.CDLL):
        if not primary.startswith('GPU-') or not 70 <= percent <= 100:
            raise ValueError('Fan experiment requires an exact UUID and 70–100 percent')
        self.lib = library_factory('libnvidia-ml.so.1')
        self.active, self.changed = False, []
        signatures = {
            'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetHandleByUUID': [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            'nvmlDeviceGetUUID': [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            'nvmlDeviceGetNumFans': [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)],
            'nvmlDeviceGetMinMaxFanSpeed': [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)],
            'nvmlDeviceGetFanControlPolicy_v2': [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            'nvmlDeviceGetTargetFanSpeed': [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
            'nvmlDeviceSetFanSpeed_v2': [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint],
            'nvmlDeviceSetDefaultFanSpeed_v2': [ctypes.c_void_p, ctypes.c_uint],
            'nvmlDeviceGetTemperature': [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self.call('nvmlInit_v2')
        self.active = True
        try:
            self.handle = ctypes.c_void_p()
            self.call('nvmlDeviceGetHandleByUUID', primary.encode('ascii'), ctypes.byref(self.handle))
            identity = ctypes.create_string_buffer(96)
            self.call('nvmlDeviceGetUUID', self.handle, identity, len(identity))
            if identity.value.decode('ascii') != primary:
                raise RuntimeError('Fan GPU identity mismatch')
            count, lower, upper = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
            self.call('nvmlDeviceGetNumFans', self.handle, ctypes.byref(count))
            self.call('nvmlDeviceGetMinMaxFanSpeed', self.handle, ctypes.byref(lower), ctypes.byref(upper))
            if not 1 <= count.value <= 8 or not lower.value <= percent <= upper.value:
                raise ValueError('Unsupported fan count or requested speed')
            self.percent, self.count = percent, count.value
            self.require_automatic()
        except BaseException:
            self.close()
            raise

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f'{name} failed with NVML status {code}')

    def value(self, name, *args):
        value = ctypes.c_uint()
        self.call(name, self.handle, *args, ctypes.byref(value))
        return value.value

    def require_automatic(self):
        for fan in range(self.count):
            if self.value('nvmlDeviceGetFanControlPolicy_v2', fan) != 0:
                raise RuntimeError('Existing manual fan policy; refusing to replace it')

    def apply(self):
        self.require_automatic()
        # Never lower the current automatic fan target for this experiment.
        for fan in range(self.count):
            if self.value('nvmlDeviceGetTargetFanSpeed', fan) > self.percent:
                raise RuntimeError('Automatic fan target exceeds experimental setting')
        for fan in range(self.count):
            self.changed.append(fan)  # restore even if a setter changes state then fails
            self.call('nvmlDeviceSetFanSpeed_v2', self.handle, fan, self.percent)
            if (self.value('nvmlDeviceGetTargetFanSpeed', fan) != self.percent or
                    self.value('nvmlDeviceGetFanControlPolicy_v2', fan) != 1):
                raise RuntimeError('Fan setting readback mismatch')

    def check(self):
        temperature = self.value('nvmlDeviceGetTemperature', 0)
        if not 0 <= temperature < 80:
            raise RuntimeError('Fan experiment reached 80C stop or invalid telemetry')
        for fan in self.changed:
            if (self.value('nvmlDeviceGetTargetFanSpeed', fan) != self.percent or
                    self.value('nvmlDeviceGetFanControlPolicy_v2', fan) != 1):
                raise RuntimeError('Fan control changed during experiment')

    def close(self):
        errors = []
        for fan in self.changed[:]:
            for attempt in range(3):
                try:
                    self.call('nvmlDeviceSetDefaultFanSpeed_v2', self.handle, fan)
                    if self.value('nvmlDeviceGetFanControlPolicy_v2', fan) != 0:
                        raise RuntimeError('Automatic fan policy was not restored')
                    self.changed.remove(fan)
                    break
                except RuntimeError as error:
                    if attempt == 2:
                        errors.append(str(error))
                    else:
                        time.sleep(0.2)
        if self.active:
            self.call('nvmlShutdown')
            self.active = False
        if errors:
            raise RuntimeError('Fan restoration failed: ' + '; '.join(errors))


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
    parser.add_argument('--fan-percent', type=int, help='Optional temporary RTX fan target, 70–100 percent')
    parser.add_argument('--fan-start-file', type=Path, help='Optional: keep automatic fans until this file appears')
    parser.add_argument('--abort-file', type=Path, help='Required for fans: tells the render controller to stop on failure/deadline')
    args = parser.parse_args()
    if not args.primary.startswith('GPU-') or not 1 <= args.seconds <= 7200:
        parser.error('Use an exact GPU UUID and duration between 1 and 7200 seconds')
    if args.stop_file.exists():
        parser.error('Stop file already exists; inspect prior experiment before proceeding')
    if args.fan_start_file and args.fan_percent is None:
        parser.error('--fan-start-file requires --fan-percent')
    if args.fan_percent is not None and (not args.abort_file or not 70 <= args.fan_percent <= 100):
        parser.error('Fan tests require --abort-file and 70–100 percent')
    paths = [p.resolve() for p in (args.stop_file, args.fan_start_file, args.abort_file) if p]
    if len(paths) != len(set(paths)) or any(p.exists() for p in paths):
        parser.error('Control files must be distinct and must not already exist')
    before = query(args.primary)
    if not before['minimum_watts'] <= args.watts < before['original_watts']:
        parser.error('Requested cap must be within the GPU limits and below its current cap')
    fans = TemporaryFans(args.primary, args.fan_percent) if args.fan_percent is not None else None
    print(json.dumps({'plan': before, 'temporary_watts': args.watts, 'temporary_fan_percent': args.fan_percent,
                      'fan_count': fans.count if fans else None, 'maximum_seconds': args.seconds,
                      'stop_file': str(args.stop_file), 'run': args.run}), flush=True)
    if not args.run:
        if fans:
            fans.close()
        return
    if os.geteuid() != 0:
        parser.error('--run requires administrator authentication')
    def interrupted(signum, frame):
        raise InterruptedError('Power experiment interrupted; restoring original cap')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with temporary_limit(args.primary, args.watts):
            try:
                end = time.monotonic() + args.seconds
                applied = False
                while time.monotonic() < end and not args.stop_file.exists():
                    if fans:
                        if not applied and (not args.fan_start_file or args.fan_start_file.exists()):
                            fans.apply()
                            applied = True
                            print(json.dumps({'phase': 'fans_applied', 'percent': args.fan_percent}), flush=True)
                        fans.check()
                    time.sleep(0.5)
                if args.abort_file and not args.stop_file.exists():
                    args.abort_file.touch()
            finally:
                if fans:
                    fans.close()
                    print(json.dumps({'phase': 'fans_restored', 'policy': 'automatic'}), flush=True)
    except BaseException:
        if args.abort_file:
            args.abort_file.touch()
        raise


if __name__ == '__main__':
    main()
