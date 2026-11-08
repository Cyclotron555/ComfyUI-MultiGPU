"""Inspect NVIDIA UUID mapping; print a launch plan unless --launch is supplied."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def select_gpu(rows, selector):
    matches = [r for r in rows if selector in (r['uuid'], r['pci.bus_id'])]
    if len(matches) != 1:
        raise ValueError(f"Expected one GPU matching UUID or PCI address {selector!r}")
    return matches[0]


def main():
    from wan_performance import launch_flags
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--primary', default='00000000:06:00.0')
    parser.add_argument('--donor', help='Optional donor UUID or PCI address; RAM only by default')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--wan-performance', action='store_true',
                        help='Opt in to a Wan BF16 UNet session with native PyTorch attention and static VRAM. Does not enable approximate attention/cache.')
    args = parser.parse_args()
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,pci.bus_id,name,memory.total,memory.used,driver_version',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(result.stdout.splitlines(), fieldnames=["uuid", "pci.bus_id", "name", "memory.total", "memory.used", "driver_version"])]
    primary = select_gpu(rows, args.primary)
    if 'RTX 3060' not in primary['name'] or int(primary['memory.total']) < 11000:
        raise ValueError('Primary does not match the expected RTX 3060 12GB; inspect hardware before proceeding')
    selected = [primary]
    if args.donor:
        donor = select_gpu(rows, args.donor)
        if donor == primary:
            raise ValueError('Primary cannot also be donor')
        selected.append(donor)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=','.join(r['uuid'] for r in selected),
               MGPU_PRIMARY_DEVICE='cuda:0', MGPU_SAFE_OFFLOAD='1')
    # UUID ordering deliberately maps the verified physical primary to logical cuda:0.
    root = args.comfy_root.resolve()
    if not (root / 'main.py').is_file():
        raise ValueError('ComfyUI main.py not found')
    command = [sys.executable, str(root / 'main.py'), '--cuda-device', 'all', '--reserve-vram', '2']
    command.extend(launch_flags(args.wan_performance))
    print(json.dumps({'physical_gpus': rows, 'logical_mapping': {f'cuda:{i}': r for i, r in enumerate(selected)},
                      'environment': {k: env[k] for k in ('CUDA_DEVICE_ORDER', 'CUDA_VISIBLE_DEVICES', 'MGPU_PRIMARY_DEVICE', 'MGPU_SAFE_OFFLOAD')},
                      'command': command, 'offload': 'cuda:1,cpu' if args.donor else 'cpu',
                      'note': 'Workflow device choices must match this mapping. No driver/power/display changes.'}, indent=2), flush=True)
    if args.launch:
        os.execve(sys.executable, command, env)


if __name__ == '__main__':
    main()
