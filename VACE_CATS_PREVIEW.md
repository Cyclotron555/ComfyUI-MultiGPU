# Guarded VACE cats preview

The current working cats preview uses **ComfyUI's native Wan2.1 VACE 1.3B model**, DisTorch2 weight placement and the existing `wan_vace_reference.prepare_conditioning` helper. A 17-frame, 288×512, 16fps test retained photographic fur and a stable courtyard without the severe neon breakup seen in the current Animate setup. Motion begins with a hold and then a paw movement; longer identity consistency, anatomy and choreography remain unvalidated. This is inference, not training.

Settings and measurements are in [the hardware profile](profiles/wan21_vace_cats_3060.json) and [measurement record](benchmarks/results/rtx3060-wan21-vace-cats.json). The source recording, images, generated video, embeddings and model weights remain local.

## Reproduce the configuration

Use an existing `wan2.1_vace_1.3B_fp16.safetensors` checkpoint, Wan2.1 VAE and Wan-compatible UMT5 text encoder. This configuration does not download models. The original checkpoint source is [Comfy-Org/Wan-2.1](https://huggingface.co/Comfy-Org/Wan-2.1/tree/main/split_files/diffusion_models); the profile records its verified SHA256.

1. Resolve physical devices by UUID/name/PCI address. Set `CUDA_VISIBLE_DEVICES` to the RTX UUID first, GTX UUID second **before importing Torch**. Only then do logical `cuda:0` and `cuda:1` refer to these roles. Verify both names at runtime.
2. Load through `UNETLoaderDisTorch2MultiGPU`, compute on the RTX, with `cuda:0,1gb;cuda:1,1gb;cpu,*` allocation. Here `gb` means GiB. GTX stores weights; it does not run the denoising model.
3. Use native `ModelSamplingSD3` shift 8 and `ModelComputeDtype` BF16. Sample 30 UniPC/simple steps, CFG 6, seed 30601660, 17 frames at 288×512. Keep a 3GiB desktop reserve.
4. Prepare native VACE reference and pose conditioning. For pre-encoded inputs, pass raw Wan VAE gray/inactive video latents, pose/reactive video latents and the single-frame reference to `wan_vace_reference.prepare_conditioning`. Its shape/finite checks and reference prefix match native `WanVaceToVideo` semantics. Preserve its returned trim count.
5. After sampling, discard the returned reference-prefix latent count before native FP32 Wan VAE decoding. Run decoding in a separate process after the sampling worker exits. Export at 16fps, without interpolation; the validated clip is silent.

The recorded sample used about 2.32GiB total RTX memory and 1.16GiB GTX memory, reached 76°C with cooling pauses, and took 124.90 seconds including model loading/guard overhead. Decode guard duration was 9.33 seconds. These are measurements of this one short clip, not a speed or quality equivalence claim against Animate.

## Keep heavy jobs separate from desktop apps

Use a systemd **user** service rather than inheriting a browser application's memory group. The new standalone launcher prints a plan unless `--run` is supplied:

```sh
python tools/wan_safe_job.py --kind check --run
python tools/wan_safe_job.py --kind render --run -- /absolute/venv/bin/python /absolute/worker.py
```

Run from the worker's intended working directory. Use absolute interpreter/script paths. The fixed `codex-wan-safe-job.service` name refuses a second simultaneous job. Existing jobs are not stopped by a refused launch.

For this 64GB-RAM machine, render stages have `MemoryHigh=36GiB`, `MemoryMax=40GiB`, swap disabled, group OOM termination and an 1800-second limit. Preparation uses 20/24GiB; downloads use 6/8GiB and two transfer streams. These are hardware-specific caps. `tools/wan_job_limits.py` lets a worker verify its isolated group before importing model libraries.

The launcher bounds CPU memory; the worker must also monitor total physical GPU use, available RAM and memory pressure, use `worker_thermal.ThermalPacer`, support an owned STOP file and terminate only its own process group. The tested controller stops above 9216MiB RTX use, 4800MiB GTX use, or 82°C, and below 8GiB available RAM. It pauses at 75°C and resumes at 67°C. No driver, display, suspend, fan or power-limit changes are required.

## Animate investigation boundary

The same cached reference/text/pose data produced corrupt Animate previews with both compressed and original weights. Full FP32 compute, independent FP32 text, alternate attention, synchronous transfers, no pose and a neutral face did not remove it. Those failures are preserved locally; no exact underlying Animate defect is claimed fixed by changing models.

The [Wan preprocessing guide](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md) recommends retargeting when source and character proportions differ, and notes limits in its single-person replacement preprocessing. Our original two-person human guides were not retargeted to feline proportions. This remains a plausible input-fit problem, not a demonstrated universal model limitation. A paired human control retained recognizable motion but still showed background/graphic artifacts. A static-background/mask test preserved known outer regions but left the regenerated cats region corrupt with visible seams. Neither is accepted as an Animate quality fix.
