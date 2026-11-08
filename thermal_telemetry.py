"""Read-only, process-local NVML temperature queries. No GPU work at import time."""
import ctypes
import os


class NvmlTemperatureReader:
    backend = 'nvml'

    def __init__(self, primary_uuid, *, library_factory=ctypes.CDLL):
        if not primary_uuid.startswith('GPU-'):
            raise ValueError('Use an exact NVIDIA GPU UUID')
        self._owner = os.getpid()
        self._active = False
        self._lib = library_factory('libnvidia-ml.so.1')
        signatures = {
            'nvmlInit_v2': [], 'nvmlShutdown': [],
            'nvmlDeviceGetHandleByUUID': [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            'nvmlDeviceGetUUID': [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint],
            'nvmlDeviceGetTemperature': [ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        }
        for name, arguments in signatures.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
        self._check(self._lib.nvmlInit_v2(), 'initialize')
        self._active = True
        try:
            self._handle = ctypes.c_void_p()
            self._check(self._lib.nvmlDeviceGetHandleByUUID(primary_uuid.encode('ascii'),
                                                         ctypes.byref(self._handle)), 'select UUID')
            identity = ctypes.create_string_buffer(96)
            self._check(self._lib.nvmlDeviceGetUUID(self._handle, identity, len(identity)), 'verify UUID')
            if identity.value.decode('ascii') != primary_uuid:
                raise RuntimeError('NVML GPU identity mismatch')
            self()  # validate live telemetry before using this reader
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(code, operation):
        if code:
            raise RuntimeError(f'NVML {operation} failed with status {code}')

    def __call__(self):
        if not self._active or os.getpid() != self._owner:
            raise RuntimeError('NVML reader is closed or belongs to another process')
        value = ctypes.c_uint()
        self._check(self._lib.nvmlDeviceGetTemperature(self._handle, 0, ctypes.byref(value)),
                    'temperature query')
        if not 0 <= value.value <= 150:
            raise RuntimeError('Invalid NVML temperature')
        return value.value

    def close(self):
        if self._active and os.getpid() == self._owner:
            self._active = False
            self._check(self._lib.nvmlShutdown(), 'shutdown')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
