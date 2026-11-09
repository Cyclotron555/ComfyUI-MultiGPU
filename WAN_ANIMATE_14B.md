# Wan 2.2 Animate 14B on RTX 3060 + GTX 1660

This experiment evaluates a larger character-animation/replacement model using the scaled FP8 checkpoint linked by the [official ComfyUI tutorial](https://github.com/Comfy-Org/docs/blob/main/tutorials/video/wan/wan2-2-animate.mdx). It is an inference setup, not a training implementation. Existing small Wan checkpoints are retained.

The four assets total 26.66 GB (decimal). Exact repository revisions, byte sizes and SHA256 hashes are in [the hardware profile](profiles/wan22_animate_14b_3060.json). Models stay local; no weights or user media are committed. The RTX 3060 has no native FP8 compute acceleration: compressed storage uses native ComfyUI casting. No acceleration LoRA is used in the baseline.

## Device and safety setup

Select the RTX by its exact UUID before importing Torch. For the optional donor trial, expose RTX first and GTX second; logical `cuda:0` then deliberately refers to RTX. Never infer the primary from NVIDIA's physical index. GTX is a weight donor, with RAM fallback; 12GB and 6GB do not become one contiguous 18GB GPU.

Keep a 3GiB core desktop reserve and independently monitor total physical memory: RTX <=9216MiB, GTX <=4800MiB, available RAM >=8GiB. Stop at 82C; worker pacing pauses at 75C and resumes at 67C. Monitor only owned process groups, support a STOP file, and bound each validation stage to 900 seconds. No suspend, display reassignment, driver, power-limit or fan changes are involved.

## Validation boundary

The completed smoke test used native text, CLIP Vision, VAE and `WanAnimateToVideo` conditioning, then nine frames at 144x256 and one Euler step. Synthetic black pose and gray face inputs exercise optional motion branches; they are not real motion guidance. Native VAE decoding checks finite output and dimensions. Such a pass establishes execution at this tiny shape only, not usable animation quality, larger-video capacity or a speed estimate for final rendering.

For real character replacement, supply a short source video and character images, prepare body/face guidance and tracked masks, and validate identity through occlusions. The official GUI template also requires additional preprocessing nodes that have not been installed by this experiment. Two-fighter replacement and one-minute generation remain untested.

Actual training requires a trainer supporting this exact architecture, a suitable dataset and a separately guarded backward-pass memory test. Inference donor placement does not establish distributed training support.

## Observed results — 2026-11-09

Both offload paths passed with finite latents; the nine decoded frames were finite at the expected dimensions. Latents were bit-identical between the native RAM baseline and GTX donor trial (maximum absolute difference zero).

| Trial | Peak total RTX memory | Peak total GTX memory | Maximum monitored temperature | Guarded stage duration |
|---|---:|---:|---:|---:|
| Native RAM offload | 9055 MiB | 2 MiB | 58C | 18.65 s |
| GTX + RAM donor | 6856 MiB | 2174 MiB | 62C | 21.85 s |

The donor trial used `cuda:0,5gb;cuda:1,2gb;cpu,*` expert allocation (`gb` is parsed in GiB). It saved RTX memory but was slower in this tiny probe; no speedup is claimed. Minimum available system RAM stayed above 41GiB. Native casting selected FP16 computation. Logical tensor byte counts can overstate compressed physical storage; use the independent monitor peaks for GPU footprint.

The larger donor test exposed a headless cleanup crash: `PromptServer.instance` does not exist until the server is initialized. Three extension cleanup entry points now treat an absent instance like `None`. Existing GUI queue behavior is preserved. All 47 CPU safety tests passed, including four new absent-server/idle/active-queue regression checks. ComfyUI core was unchanged. The installed extension was backed up before the fix.

See [the full measurement record](benchmarks/results/rtx3060-wan22-animate14b.json). These results do not establish training support or final karate-video quality.

## Real cats preview status

The current two-cat Animate setup failed visual review with scaled FP8 and original BF16 weights. Removing pose, adding a neutral face, replacing text embeddings with existing FP32 UMT5 output, using full FP32 model computation, changing attention implementations, and disabling asynchronous offloading did not remove corruption. Finite output is not a quality pass. A paired human control still showed background/graphic artifacts, and a protected-background test left the regenerated region corrupt. An exact Animate defect has not been established.

The existing VACE 1.3B model produced a clean short preview from the same cached reference, text and pose data. Use the [VACE preview notes](VACE_CATS_PREVIEW.md) for that validated configuration. Body-motion retargeting and longer choreography remain open work; the [Wan preprocessing guide](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/animate/preprocess/UserGuider.md) recommends retargeting when character proportions differ.
