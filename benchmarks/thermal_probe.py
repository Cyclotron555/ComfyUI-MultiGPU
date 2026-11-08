"""Read-only comparison of live NVML and subprocess temperature-query overhead."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thermal_telemetry import NvmlTemperatureReader
from worker_thermal import read_temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', required=True, help='Exact NVIDIA GPU UUID')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1000:
        parser.error('Iterations must be between 1 and 1000')
    result = {'primary_uuid': args.primary, 'read_only': True,
              'iterations': args.iterations, 'note': 'Query latency, not model speed or thermal performance'}
    with NvmlTemperatureReader(args.primary) as nvml:
        for name, reader in [('nvml', nvml), ('smi', lambda: read_temperature(args.primary))]:
            reader()  # exclude initialization / first call
            start = time.perf_counter()
            values = [reader() for _ in range(args.iterations)]
            result[name] = {'seconds_per_query': (time.perf_counter() - start) / args.iterations,
                            'minimum_temperature_c': min(values), 'maximum_temperature_c': max(values)}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        with args.output.open('x') as file:
            file.write(text)
    print(text)


if __name__ == '__main__':
    main()
