"""Bounded, opt-in probe of core Canny preprocessing; never loads model weights."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=Path, required=True)
    parser.add_argument('--primary', required=True, help='Exact RTX3060 GPU UUID')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--frames', type=int, default=17)
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.primary.startswith('GPU-') or min(args.frames, args.width, args.height, args.batch_size) < 1:
        parser.error('Use an exact UUID and positive dimensions/counts')
    paths = [args.source_dir / f'{i:03}.png' for i in range(args.frames)]
    if not all(p.is_file() for p in paths):
        parser.error('Source frame sequence is incomplete')
    print(json.dumps({'run': args.run, 'frames': args.frames, 'primary': args.primary,
                      'operation': 'NativeCanny0.4/0.8', 'model_downloads': False}), flush=True)
    if not args.run:
        return
    if args.output_dir.exists():
        parser.error('Output directory already exists; preserve prior results')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.primary
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    sys.path.insert(0, str(args.comfy_root.resolve()))
    # The optional GPU probe must set visibility/options before importing core.
    import comfy.options
    comfy.options.enable_args_parsing()
    sys.argv = [sys.argv[0], '--offline', '--reserve-vram', '2']
    import torch
    import numpy as np
    from PIL import Image
    from comfy_extras.nodes_canny import Canny
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from worker_thermal import ThermalPacer
    if 'RTX 3060' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Selected UUID is not the expected RTX3060')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    pacer = ThermalPacer(args.primary, fence=lambda: torch.cuda.current_stream(0).synchronize(),
                         drain=lambda: torch.cuda.synchronize(0), telemetry_backend='nvml')
    args.output_dir.mkdir()
    torch.cuda.reset_peak_memory_stats(0)
    start = time.monotonic()
    changed_pixels = 0
    try:
        for offset in range(0, len(paths), args.batch_size):
            pacer.check()
            images = []
            for path in paths[offset:offset + args.batch_size]:
                with Image.open(path) as image:
                    images.append(np.array(image.convert('RGB').resize((args.width, args.height), Image.Resampling.LANCZOS)))
            pixels = torch.from_numpy(np.stack(images)).float().div(255)
            edges = Canny.execute(pixels, 0.4, 0.8)[0].mul(255).byte().cpu().numpy()
            if torch.cuda.mem_get_info(0)[0] < 2 * 1024**3:
                raise RuntimeError('Desktop GPU free memory fell below2GiB')
            for index, edge in enumerate(edges, offset):
                Image.fromarray(edge).save(args.output_dir / f'{index:03}.png')
                if args.reference_dir:
                    with Image.open(args.reference_dir / f'{index:03}.png') as reference:
                        changed_pixels += int(np.any(edge != np.asarray(reference.convert('RGB')), axis=-1).sum())
    finally:
        pacer.close()
    report = {'seconds': time.monotonic() - start, 'frames': args.frames,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
              'maximum_guard_c': pacer.maximum_observed_c, 'cooling_seconds': pacer.cooling_seconds,
              'reference_compared': args.reference_dir is not None, 'changed_pixels': changed_pixels,
              'scope': 'Preprocessing only; CPU serial vs GPU batched. Not generation speed.'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if args.reference_dir and changed_pixels:
        raise SystemExit('GPU/CPU controls differ; do not claim pixel identity')


if __name__ == '__main__':
    main()
