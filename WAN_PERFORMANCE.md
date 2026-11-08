# Tested Wan performance options for this fork

This fork adds opt-in orchestration around existing ComfyUI features. The attention kernels and EasyCache implementation belong to ComfyUI/Comfy Kitchen and their contributors. DisTorch2 and the original extension are by pollockjj and contributors; this fork is not an upstream endorsement.

Test machine: RTX 3060 12 GB at PCI `06:00.0` for generation/display, GTX 1660 6 GB at PCI `05:00.0` for offloaded weights, 64 GB RAM. Never copy another machine's CUDA indices. Tested core commit: `0decb39ee5b5e17c200334589fc317dbbe98310e`, Torch `2.15.0+cu132`, Comfy Kitchen `0.2.41`.

## What changed

- `safe_launch.py --wan-performance`: an explicitly selected Wan session adds native BF16 UNet execution, native PyTorch attention and the tested static-VRAM path. UUID mapping and the 2 GiB desktop reservation remain in force. Omitting the flag leaves the prior launch command unchanged.
- `wan_performance.configure_wan`: selects native PyTorch or optional native Comfy Kitchen INT8 attention through a cloned model patcher. Optional native EasyCache is off by default. Unsupported optional components return an explicit fallback report; patching errors propagate. No global attention monkeypatch, new node duplicate, downloads, driver changes or original checkpoint rewriting.
- `benchmarks/attention_probe.py`: an opt-in, model-free comparison with fixed small shapes, identity verification, memory/temperature checks and recorded numerical differences. A dry run does not import Torch or initialize CUDA.
- `worker_thermal.ThermalPacer`: optional pacing for an owned experiment worker, with injected compute-stream fence/full-device drain, bounded cool-down, exact GPU UUID telemetry and fail-closed cutoff handling. It is not automatically attached to GUI models and does not replace the outer memory/RAM/process monitor.
- `benchmarks/watch_render_power.py`: an unprivileged power trace for the comparison controller's atomic JSON state; signals the authenticated helper to restore the original cap on complete/failure or watcher exit. It never runs models or changes a GPU setting itself.

INT8 attention, BF16 compute and caching can change results. Synthetic numerical differences are not video-quality scores. Keep a reproducible baseline and compare actual motion and geometry before choosing an accelerated profile.

## Start a mapped BF16 Wan session

Use the ComfyUI environment's Python. Replace the placeholders with the exact UUIDs printed by `nvidia-smi -L`:

```bash
/path/to/ComfyUI/.venv/bin/python safe_launch.py \
  --comfy-root /path/to/ComfyUI \
  --primary GPU-RTX-UUID --donor GPU-GTX-UUID --wan-performance
```

Inspect the dry-run mapping, then add `--launch` to start. The selected physical RTX becomes logical `cuda:0` deliberately; it is not assumed to be NVIDIA-SMI GPU 0. This launcher still checks for the expected RTX 3060 12 GB. Donor selection is optional and defaults to RAM.

In a native ComfyUI workflow, use **Model Attention Backend → comfy kitchen attention** after loading the Wan model. Keep text encoding and VAE on their tested paths. The native node falls back to PyTorch when the requested backend is unavailable. EasyCache, if used, goes after the model patches; begin with a conservative threshold and preserve the dense early/late sampling window. It is an approximation, not free exact reuse.

For a scripted native ComfyUI run with this extension directory on `sys.path`:

```python
from wan_performance import configure_wan

model, report = configure_wan(
    model,
    attention="kitchen_int8",  # "pytorch" is the default
    cache_threshold=0.0,      # caching disabled; opt in only after a comparison
)
print(report)
```

This helper uses the model's existing compute device. It does not choose a GPU, set precision or perform thermal management. The caller retains its existing monitored worker and stop controls. Do not stack unrelated attention/cache patches and assume their benefits combine.

The [tested quality recipe](profiles/wan13b_3060_gtx1660.json) keeps caching off and sets the existing DisTorch loader's `virtual_vram_gb` to 0.25. This is a workflow input, not a new global memory-management rule. On this machine, repeated transfer from a larger GTX allocation cost more time despite substantial free RTX memory. A smaller allocation still uses both cards. Do not apply this amount blindly to larger models or assume that unused donor VRAM is a performance problem.

For a separate monitored worker, the pacing callbacks use its already verified compute device:

```python
from worker_thermal import ThermalPacer

pacer = ThermalPacer(
    primary_uuid,
    fence=lambda: torch.cuda.current_stream(compute_device).synchronize(),
    drain=lambda: torch.cuda.synchronize(compute_device),
)
# Invoke at bounded work boundaries, e.g. between transformer/decoder chunks.
pacer.check()
```

Keep independent outer cancellation, VRAM/RAM limits and a worker timeout. Telemetry failure, temperature cutoff or failure to cool raises an error. There is no claimed speed gain from stream fencing alone; it preserves room for prefetch overlap without changing the cooling targets.

## Evidence and interpretation

The matched short BF16 check at 768×432, 17 frames, 30 UniPC/simple steps sampled in 66.110 seconds versus 290.591 seconds with the preserved FP32 setup, using RAM offload. This one pair demonstrates a practical gain on this machine, not a universal multiplier. Both sampled and decoded tensors were finite; checkpoint files remained unchanged.

The small native-kernel test used BF16, batch 2, 12 heads, head dimension 128 and included quantization overhead. At 16,384 tokens, PyTorch Flash SDPA took 133.043 ms and native Kitchen INT8 took 51.817 ms (~2.57× kernel speed). Random-tensor relative RMS difference was ~0.0167. Peak Torch allocation was about 1.03 GiB; observed GPU temperature stayed at or below 56°C. These timings exclude model loading, layer computation, donor transfers, cooling and VAE export. Full-video measurements must be reported separately.

The installed sparse SOL kernel was also probed but showed a much larger random-tensor discrepancy. It is not selected by this helper. SageAttention and Torch compilation remain research alternatives; no new package was installed or incompatible CUDA build attempted. The native kernel already available here is the lower-change first candidate.

Completed matched video results, when available, are recorded in `benchmarks/results/` and `HARDWARE_CHANGELOG.md`. Long-shot tests are separate from the small-kernel comparison; a 161-frame shape test does not establish ten-second visual consistency.

### Matched video results on this machine

Same 768×432, 17 frames, 30 UniPC/simple steps, BF16 UNet, seed/prompt/CFG/shift and both-card mapping. All cases passed finite sample/decode and MP4 export. These are single sequential runs, with different initial thermal states; they are not statistical estimates.

| Configuration | Sampling, including cooling | Load/sample + decode/export stages |
| --- | ---: | ---: |
| PyTorch SDPA, 1 GiB donor | 102.54 s | 132.11 s |
| Native INT8, 1 GiB donor | 98.13 s | 130.17 s |
| Native INT8, 0.25 GiB donor | 78.74 s | 109.80 s |
| Native INT8, 1 GiB donor, EasyCache 0.10 | 54.28 s | 83.34 s |
| Native INT8, 0.25 GiB donor, compute-stream thermal fence | 88.83 s | 119.88 s |
| Native INT8, 0.25 GiB donor, EasyCache 0.05, reusable pacer | 53.03 s | 83.35 s |

The 1 GiB → 0.25 GiB donor change preserved native INT8 sampled latents **bit-for-bit** in this pair while reducing sampling time by about 20%. It removes transfer work rather than adding a new approximation. The system reported a PHB interconnect and direct peer reads unsupported between these GPUs. This does not make maximum donor occupancy a useful speed target.

The synchronization variants were also bit-identical. Their time excluding cooling was similar (~72–73 s); no independent stream-fence speed gain was established. EasyCache 0.10 skipped 12/30 steps; 0.05 skipped 9/30. Both changed output geometry/composition, so caching stays off in the quality recipe. Lower sampling time is not evidence of equal video quality.

Short-run peak sampled VRAM stayed below 4,245 MiB on the RTX and 1,179 MiB on the GTX; sampled temperatures stayed at or below 75°C. Minimum available RAM remained above 50 GiB. See [the complete machine-readable report](benchmarks/results/rtx3060-wan-video.json), which includes each case's actual placement, cooling time and limitations. No new package, model, driver or system power change was required.

### Full-resolution confirmation, caching off

A matched 832×480, 53-frame (3.3125-second) pair at the original 170 W cap sampled in **844.31 → 421.24 seconds** (~2.00× sampling speed). The sum of load/sample and decode/export stages fell **961.03 → 528.35 seconds** (~1.82×). Both used the same seed, prompt, BF16 precision, 30-step UniPC/simple schedule and both-card mapping. The new profile combines INT8 attention, the smaller donor allocation and precise cooling pauses; do not attribute the whole gain to one kernel.

Both completed finite sampling and FP32 VAE decoding. Frames show similar composition with numerical/geometry differences from INT8; no quality-equivalence claim. Peak sampled RTX usage rose from 5,302 to 6,145 MiB while GTX usage fell from 1,178 to 354 MiB, staying within reserves. Sampled maximum temperature was 76°C in both. [Full-resolution report](benchmarks/results/rtx3060-wan-highres.json).

### Separate, explicitly authorized power experiment

`benchmarks/temporary_power_limit.py` is a standalone administrator utility, **not** part of ComfyUI startup or the quality recipe. The user separately approved testing 130 W versus the board's original 170 W limit. Its results are independent of the software-only tables above.

It defaults to a read-only plan. With `--run`, it accepts only a lower cap within the reported hardware limits, verifies the exact RTX 3060 12 GB UUID, and restores the original limit in `finally`, on SIGINT/SIGTERM, or after a bounded timer (default 900 s; explicit maximum 7200 s for separately authorized long renders). Restoration has readback and retries. Model code is never run as root. No fan, manual clock, driver or display changes are made. Forced termination cannot run Python cleanup; verify the actual cap after any interruption.

Inspect the helper's dry-run plan first. When explicitly choosing this experiment, authenticate for the isolated system-Python helper only; run the comparison worker normally in another process:

```bash
pkexec /usr/bin/python3 -I /absolute/path/benchmarks/temporary_power_limit.py \
  --primary GPU-RTX-UUID --watts 130 --seconds 900 \
  --stop-file /tmp/my-wan-power-test-done --run
```

Creating the chosen stop file ends the experiment and restores the original cap. An already-existing stop file is rejected before any write. The NVIDIA [power-limit documentation](https://docs.nvidia.com/deploy/nvidia-smi/) describes its administrator requirement and allowed hardware range. This experiment is not an overclock and is not automatically recommended as faster.

The completed matched 130 W test sampled in 356.61 s versus 421.24 s at 170 W (~15% shorter), with **bit-identical latents, all 53 PNG frames and MP4 bytes**. Stage wall time fell from 528.35 to 452.77 s (~14% shorter). The administrator helper restored 170 W and readback verified it before any adoption decision. This is one matched pair, not a universal power-efficiency claim. See [the power comparison report](benchmarks/results/rtx3060-power-comparison.json).

The user separately authorized using 130 W for the queued videos, with restoration after completion and a two-hour fallback deadline. An independent user-process watcher records actual power settings and signals restoration on controller completion/failure. Its complete/failed-state cleanup paths were exercised without GPU access. Power limits are still not changed automatically by the quality recipe, and the generic watcher requires the comparison controller's documented `status` state rather than arbitrary ComfyUI queue data.

## Reproduce the small kernel probe

```bash
/path/to/ComfyUI/.venv/bin/python benchmarks/attention_probe.py \
  --primary GPU-RTX-UUID
# Add --run --output attention-results.json for the actual bounded probe.
```

Run it while no video render is active. Its 75°C stop is intentionally below the generation workers' 82°C cutoff. Unsupported optional backends are recorded as unavailable. No result is substituted for an unexecuted test.

## Validation and rollback

CPU-only tests: `python -m unittest discover -s tests -v` (42 passed, including the cooling additions). The new tests cover clone isolation, nonzero compute-device mapping, unavailable native kernels, missing optional cache, invalid cache windows, unchanged baseline launch flags, thermal cutoff/timeout handling, restoring the original power cap on completion/interruption and a read-only long-render plan. Actual finite-tensor/MP4 and visual checks complement those tests.

Remove `--wan-performance` to restore the prior launch flags. Select PyTorch attention and remove/bypass EasyCache to restore baseline model patches. Original FP32 weights and earlier videos remain separate. This work does not fix or retest the known NVIDIA suspend/resume issue.

## Primary sources

- [ComfyUI Model Attention Backend and compute-dtype nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_model_advanced.py)
- [ComfyUI native EasyCache](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_easycache.py)
- [Comfy Kitchen](https://github.com/Comfy-Org/comfy-kitchen)
- [PyTorch SDPA and backend selection](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [SageAttention hardware/build requirements](https://github.com/thu-ml/SageAttention)
- [TeaCache's Wan2.1 experiments](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4Wan2.1) (A800 figures are not RTX 3060 results; its thresholds are not interchangeable with EasyCache)


## Completed ten-second comparison — 2026-11-08

Both native 832×480, 16 fps videos completed with finite sampling and decoding: continuous 161 frames (10.062 s), and three independent 53-frame shots (159 frames, 9.938 s). No upscaling or frame interpolation was used. Sampled first/middle/last frames show a consistent boat and horizon in the continuous clip; the independent shots vary in boat design and lighting. This frame inspection does not establish perfect temporal consistency.

The continuous run sampled in 2,196.45 s, decoded in 277.70 s, and took 2,488.32 s across measured stages (41.47 minutes). Cooling accounted for 920.82 s of sampling, so sustained cooling remains a major bottleneck. Peak RTX allocation reported by monitoring was 7,630 MiB at 75°C; GTX peak was 354 MiB at 47°C, with at least 50.84 GiB RAM available. The RTX remained compute/display primary; the GTX stored offloaded weights rather than running parallel denoising. Native INT8 attention, BF16 transformer, FP32 VAE and 0.25 GiB donor settings were used, with approximate step caching disabled.

The explicitly approved temporary 130 W limit was observed throughout the monitored batch. After completion, the administrator helper restored the original 170 W RTX limit; independent NVIDIA readback confirmed it. All owned render workers, the watcher and power helper exited successfully. No matching current-boot NVIDIA Xid/assertion or blocked-task entries were found in the journal check. Suspend/resume was not exercised or fixed. Existing originals remain preserved.

See [the completed measurement report](benchmarks/results/rtx3060-ten-seconds.json). The prior 30 CPU-only tests passed; this final update changes documentation and recorded results only. The small 1.3B model and extended sequence length still limit fidelity and temporal reliability; these results are not a general performance guarantee.


## Cooling experiments and direct temperature reads

`ThermalPacer(..., telemetry_backend="nvml")` opts into process-local, read-only NVIDIA NVML temperature queries. The default stays `smi` for compatibility. Both readers retain the same compute-stream fence, full drain before cooling, 75°C pause, 67°C resume and 82°C cutoff. NVML selects and verifies the exact GPU UUID, reads live values without caching, rejects use after fork/close, and stops on driver/query failure. Call `pacer.close()` when done. Importing the module does not initialize the driver; no extra Python package is needed. There is no silent backend substitution when NVML was explicitly requested.

Read-only latency probe (this measures telemetry overhead, not generation speed):

```bash
python benchmarks/thermal_probe.py --primary GPU-RTX-UUID --iterations 100
```

The existing bounded power helper now also supports an **optional** temporary 70–100% RTX fan target through NVML. It requires the exact RTX3060 12GB UUID, rejects existing manual fan policies and refuses to lower an existing automatic fan target. Every changed fan is restored to automatic, including on partial setter failure. An independent 80°C helper stop and the worker's existing monitoring remain active. A deadline or error touches an abort file that the experiment controller must watch and use to stop its own worker. Fans and power are not changed by normal extension import, launch flags or recipe loading.

```bash
# Read-only plan first; create a fresh control directory and use absolute paths.
python benchmarks/temporary_power_limit.py \
  --primary GPU-RTX-UUID --watts 130 --seconds 7200 \
  --fan-percent 75 --abort-file /absolute/path/ABORT \
  --stop-file /absolute/path/POWER_DONE
# For an explicitly authorized actual test, run the same arguments using:
# pkexec /usr/bin/python3 -I /absolute/path/benchmarks/temporary_power_limit.py ... --run
```

An optional `--fan-start-file /absolute/path/FAN_ON` leaves automatic fan control active until that file appears, allowing a matched baseline first. Control files must be distinct and absent initially. Creating the stop file restores automatic fans and the original power cap. A controller must also monitor resources, stop its owned render on abort and create the stop file on exit. The helper has a two-hour maximum fallback; it is not a permanent fan service and cannot recover from SIGKILL or power loss. Restoration must be independently checked afterward. No drivers, display assignments, clock offsets, temperature limits, persistence mode or Xorg configuration are changed.

See the [walking-human recipe](profiles/wan13b_human_walking_3060_gtx1660.json). Exact latent and PNG comparisons, not unchanged recipe settings alone, are required before claiming output preservation. A single matched pair remains sensitive to foreground desktop activity and starting temperature.

Primary references: [NVIDIA NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html), [initialization and cleanup](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlInitializationAndCleanup.html), and [temporary fan control and restoring defaults](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceCommands.html).


## Measured cooling results on this RTX3060

All four53-frame832×480 runs used the same walking prompt,seed,30UniPC/simple steps,CFG5/shift8,BF16 transformer,FP32 VAE,nativeINT8 attention,cache off and0.25GiB GTX donor. Power was temporarily130W for every comparison.

| Worker temperature reader / pause-resume | RTX fan policy | Sampling seconds | Cooling wait seconds |
| --- | --- | ---: | ---: |
| SMI /75–67°C reference | Automatic |318.92|47.55|
| NVML /75–67°C |75% temporary |355.99|97.80|
| NVML /75–67°C |90% temporary |315.66|57.28|
| NVML /77–72°C pilot |90% temporary |280.03|19.26|

The last pilot reduced sampling time by **12.19%**, cooling waits by **59.49%**, and measured sampling-plus-decode stage wall time by **8.47%**. Latent files, **all53PNG frames and MP4 bytes were identical** to the reference for every candidate. The75% fan candidate was slower and rejected. The old75°C guard plus90% fan result changed by only about1%, too small for a reliable performance claim. Reducing query latency alone does not guarantee faster generation: eliminating CPU gaps increases sustained GPU load and can create additional cooling waits.

The successful pilot changes only the worker's software pause/resume points to77/72°C. The extension default remains75/67°C. The worker82°C cutoff, independent helper80°C abort, desktop VRAM reserve and RAM/donor guards remain active; VAE decoding still uses75/67°C. The local driver reports an83°C target and93°C maximum operating temperature; no NVIDIA temperature limit is changed. The monitored pilot reached77°C and completed finite sampling/decoding. Both fans were independently verified back in automatic mode and the RTX back at170W afterward.

This is one sequential pair per candidate, with possible differences in initial temperature,desktop activity and case airflow. The result is for combined fan,telemetry and pacing changes on this hardware, not an isolated fan effect, universal speedup or proof of long-run stability. No suspend/resume or driver changes were tested. See [full measurements, negative results and restoration readback](benchmarks/results/rtx3060-cooling-walking.json).42CPU-only tests passed, including explicit77/72°C pacing and unchanged82°C cutoff.

The anime recipe is a separate creative test. Its motion-first prompt andCFG6/shift8 follow the [official Wan1.3B example](https://github.com/Wan-Video/Wan2.1/blob/main/README.md), with30sampling steps retained for the intended final render. The first short anime style preview achieved the2D look but insufficient walking motion; it is not a completed10-second walking result. Motion is reviewed before continuing a costly longer shot. No new weights or approximation cache were introduced.
