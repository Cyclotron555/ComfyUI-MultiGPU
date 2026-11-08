"""Small opt-in attention comparison; no models, installs or system changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def snapshot(primary):
    result = subprocess.check_output([
        'nvidia-smi', '--id=' + primary,
        '--query-gpu=uuid,name,memory.total,memory.used,temperature.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=10)
    identity, name, total, used, temperature = [value.strip() for value in result.split(',')]
    if identity != primary or 'RTX 3060' not in name:
        raise ValueError('Expected the configured RTX 3060 UUID; no index-based substitute allowed')
    if int(used) > int(total) - 2048 or int(temperature) >= 75:
        raise RuntimeError('Probe stopped: desktop memory reserve or temperature limit reached')
    return {'uuid': identity, 'name': name, 'used_mib': int(used), 'temperature_c': int(temperature)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', required=True, help='Exact RTX 3060 GPU UUID')
    parser.add_argument('--run', action='store_true', help='Actually run; otherwise print the plan')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.primary.startswith('GPU-'):
        parser.error('--primary must be a GPU UUID, not a CUDA or nvidia-smi index')
    report = {'primary': args.primary, 'tokens': [4096, 16384], 'batch': 2, 'heads': 12,
              'head_dim': 128, 'dtype': 'bfloat16', 'repeats': 3,
              'note': 'Synthetic kernel timing including quantization; not whole-video speed or quality.'}
    if not args.run:
        print(json.dumps(report, indent=2))
        return
    report['before'] = snapshot(args.primary)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.primary
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    import torch
    import comfy_kitchen as kitchen
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn import functional as functional
    if torch.cuda.device_count() != 1 or 'RTX 3060' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Primary mapping did not produce the expected RTX 3060')
    report.update(torch=torch.__version__, cuda=torch.version.cuda,
                  capability=list(torch.cuda.get_device_capability(0)), results=[])
    torch.manual_seed(30601660)
    with torch.inference_mode():
        for tokens in report['tokens']:
            tensors = [torch.randn((2, tokens, 12, 128), device='cuda:0', dtype=torch.bfloat16).transpose(1, 2) for _ in range(3)]
            def sdpa():
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                                  SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True):
                    return functional.scaled_dot_product_attention(*tensors)
            backends = {'pytorch_sdpa': sdpa}
            if kitchen.int8_attention_is_available(torch.device('cuda:0')):
                backends['kitchen_int8'] = lambda: kitchen.int8_attention(*tensors)
            if kitchen.sol_attn_is_available(torch.device('cuda:0')):
                backends['sol_tau_0.5'] = lambda: kitchen.sol_attn(
                    *(tensor.transpose(1, 2) for tensor in tensors), tau=0.5).transpose(1, 2)
            reference = sdpa()
            for name, function in backends.items():
                snapshot(args.primary)
                row = {'backend': name, 'tokens': tokens}
                try:
                    output = function()
                    torch.cuda.synchronize()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(3):
                        output = function()
                    end.record()
                    torch.cuda.synchronize()
                    difference = output.float() - reference.float()
                    row.update(milliseconds=begin.elapsed_time(end) / 3,
                               finite=bool(torch.isfinite(output).all()),
                               relative_rms_error=float(difference.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp_min(1e-6)),
                               max_abs_error=float(difference.abs().max()))
                    del output, difference
                except (RuntimeError, AttributeError, TypeError) as error:
                    row['unavailable'] = str(error)
                report['results'].append(row)
                report['after_latest'] = snapshot(args.primary)
                time.sleep(2)
            if tokens == 4096:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                    sdpa()
                    torch.cuda.synchronize()
                report['sdpa_dispatch_operators'] = [event.key for event in profiler.key_averages() if 'attention' in event.key]
            del tensors, reference
    report['peak_torch_bytes'] = torch.cuda.max_memory_allocated()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
