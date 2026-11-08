"""Optional process-local pacing for monitored experiment workers, not system tuning.

The caller supplies a compute-stream fence and a full-device drain. Cold checks
do not drain prefetch streams. The outer worker monitor remains responsible for
VRAM, RAM, other GPUs, cancellation and killing a stuck process.
"""
import math
import subprocess
import time


def read_temperature(primary_uuid):
    if not primary_uuid.startswith('GPU-'):
        raise ValueError('Use an exact NVIDIA GPU UUID, not a device index')
    line = subprocess.check_output([
        'nvidia-smi', '--id=' + primary_uuid, '--query-gpu=uuid,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10).strip()
    identity, temperature = [value.strip() for value in line.split(',')]
    if identity != primary_uuid:
        raise RuntimeError('GPU identity mismatch; thermal pacing stopped')
    return int(temperature)


class ThermalPacer:
    """One worker's bounded cooling pauses; no Torch import or global state."""
    def __init__(self, primary_uuid, *, fence, drain, pause_at=75, resume_at=67,
                 cutoff=82, poll_seconds=0.25, timeout_seconds=120,
                 temperature_reader=None, clock=time.monotonic, sleep=time.sleep,
                 telemetry_backend='smi'):
        values = (pause_at, resume_at, cutoff, poll_seconds, timeout_seconds)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Thermal settings must be finite')
        if not 0 <= resume_at < pause_at < cutoff or poll_seconds <= 0 or timeout_seconds <= 0:
            raise ValueError('Require resume < pause < cutoff and positive timing limits')
        if not primary_uuid.startswith('GPU-'):
            raise ValueError('Use an exact NVIDIA GPU UUID')
        if telemetry_backend not in ('smi', 'nvml'):
            raise ValueError('Telemetry backend must be smi or nvml')
        self.fence, self.drain = fence, drain
        self._owned_reader = None
        self.telemetry_backend = 'injected' if temperature_reader is not None else telemetry_backend
        if temperature_reader is not None:
            self.read = temperature_reader
        elif telemetry_backend == 'nvml':
            from thermal_telemetry import NvmlTemperatureReader
            self._owned_reader = NvmlTemperatureReader(primary_uuid)
            self.read = self._owned_reader
        else:
            self.read = lambda: read_temperature(primary_uuid)
        self.pause_at, self.resume_at, self.cutoff = pause_at, resume_at, cutoff
        self.poll, self.timeout = poll_seconds, timeout_seconds
        self.clock, self.sleep = clock, sleep
        self.cooling_seconds = 0.0
        self.pauses = 0
        self.maximum_observed_c = 0

    def close(self):
        if self._owned_reader is not None:
            self._owned_reader.close()

    def temperature(self):
        temperature = self.read()
        if not math.isfinite(temperature) or temperature < 0:
            raise RuntimeError('Invalid GPU temperature; pacing stopped')
        self.maximum_observed_c = max(self.maximum_observed_c, temperature)
        if temperature >= self.cutoff:
            raise RuntimeError(f'GPU reached the {self.cutoff}C cutoff')
        return temperature

    def check(self):
        self.fence()  # compute stream only; leave future weight prefetch overlapped
        temperature = self.temperature()
        if temperature < self.pause_at:
            return
        self.drain()  # fully idle this GPU before a cooling pause
        temperature = self.temperature()
        start = self.clock()
        while temperature > self.resume_at:
            if self.clock() - start >= self.timeout:
                raise RuntimeError('GPU did not cool within the worker timeout')
            self.sleep(self.poll)
            temperature = self.temperature()
        self.cooling_seconds += self.clock() - start
        self.pauses += 1
