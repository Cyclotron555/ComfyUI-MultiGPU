# RTX 3060 / GTX 1660 engineering log

## 2026-11-08 — initial customization

Base: Cyclotron555/ComfyUI-MultiGPU `37a73feda5804045e4f39d8c640905d62a2169d1` (same commit as installed extension).
Branch: `hardware/3060-primary-safe-offload`. Original installed extension and ComfyUI are untouched. No upstream submission or main-branch change.
ComfyUI inspected: `0decb39ee5b5e17c200334589fc317dbbe98310e`; Python environment reports PyTorch `2.15.0+cu132`.

### Observations

- NVIDIA 595.99.02: nvidia-smi GPU 0 is GTX 1660 6144 MiB, PCI 05:00.0; GPU 1 is RTX 3060 12288 MiB, PCI 06:00.0, desktop GPU.
- Actual PyTorch enumeration in the current environment is reversed: logical cuda:0 is RTX 3060, cuda:1 is GTX 1660. Never infer CUDA identity from nvidia-smi index.
- RAM: approximately 60 GiB reported by OS (64 GB installed), about 55 GiB available during inspection; 8 GiB swap.
- Only MultiGPU is installed among custom-node packages; WanVideoWrapper is absent. Its live integration cannot be verified here.
- Previous boot: repeated GPU1 PM assertions at 11:21:11, nvidia-sleep.sh blocked for 121+ seconds at 11:23:17, then 244+ seconds. NVIDIA suspend/resume/hibernate services enabled; PreserveVideoMemoryAllocations=1, TemporaryFilePath=/var. This supports a power-management hang, not a proven root cause or repair. No suspend/resume test or settings change performed.
- PCIe 2.0 x4 donor bandwidth is user-reported, not benchmarked. Offloading may be slower than RAM for some workloads.

### Exact changes and rationale

- `device_utils.py`: add default-device helper using validated MGPU_PRIMARY_DEVICE override, otherwise ComfyUI's actual selected device. Preserve device enumeration and workflow selections.
- `wanvideo.py`, `wrappers.py`: use helper for new node defaults instead of devices[1]. Saved workflows retain explicit device values and must be reviewed when mapping changes.
- `distorch_2.py`: expert allocation compute fallback uses model_patcher.load_device rather than hard-coded cuda:0. Reject negative/nonfinite virtual VRAM, unavailable/self donors; deduplicate donors. With MGPU_SAFE_OFFLOAD=1, cap virtual donations to current free-memory budgets, append CPU fallback, fail when insufficient, and check final block assignments (including tiny blocks and CLIP heads) against budgets before placement. Expert allocations receive final checks but are not automatically rewritten.
- `hardware_profile.py`: opt-in budgets use min(total, free) minus 2 GiB for configured primary, 1 GiB for other GPUs, 8 GiB for RAM. Environment overrides MGPU_GPU_RESERVE_GIB (all GPUs) and MGPU_RAM_RESERVE_GIB are finite/nonnegative. These storage checks complement ComfyUI inference headroom; they do not guarantee against OOM.
- `safe_launch.py`: read-only NVIDIA UUID/PCI diagnostics, expected RTX 3060 12GB validation, explicit process-local UUID ordering, default RAM-only offload. Optional donor enabled explicitly. Prints plan unless --launch is given. Passes --cuda-device all and --reserve-vram 2 to inspected ComfyUI; avoids its --default-device reordering. Uses invoking Python interpreter. Does not alter system GPU indices, display routing, drivers, or power settings.
- `tests/`: CPU-only regression tests, including actual DisTorch functions extracted via AST to avoid importing/patching live ComfyUI. Not an end-to-end integration test.

### Verification

10 unittest cases pass: core-primary selection, explicit/missing mapping, free-memory reservations, invalid reserves, opt-in behavior, UUID/PCI selection, GPU donor cap with RAM fallback, exhausted capacity, malformed/self donors, expert assignment guard.
Python compilation and git diff whitespace checks pass. UUID-ordered PyTorch property probe confirms RTX 3060 logical cuda:0 and GTX 1660 logical cuda:1; no tensor allocation or generation benchmark.
Launcher dry-run exercised against both actual cards. Initial dry-run exposed NVIDIA CSV headers containing units; fixed with explicit noheader field names and repeated successfully.

### Use and rollback

Install this branch as the MultiGPU extension only after backing up the existing custom_nodes/ComfyUI-MultiGPU directory. This session has NOT installed it. Run commands with the ComfyUI environment's Python, from the extension directory:

```bash
/home/claudebutnaru/AI/ComfyUI/.venv/bin/python safe_launch.py --comfy-root /home/claudebutnaru/AI/ComfyUI
# Optional GTX donor, still a dry-run:
/home/claudebutnaru/AI/ComfyUI/.venv/bin/python safe_launch.py --comfy-root /home/claudebutnaru/AI/ComfyUI --donor 00000000:05:00.0
```

Only add `--launch` after reviewing printed mapping and workflow devices. With this launcher: compute cuda:0 = verified RTX 3060; optional donor cuda:1 = GTX 1660, followed by cpu. The offload field printed by launcher is guidance, not a workflow rewrite. Select donor_device in workflow or configure the existing allocation string. Native Wan should use native loaders with MultiGPU wrappers; WanVideoWrapper requires a separate compatible install. Begin with CPU donor. No dtype, quantization, resolution, frame count, sampler or precision changes are made.

To test: `python3 -m unittest discover -s tests -v`.
Rollback: restore original extension directory or return to recorded base commit; stop using launcher. Local installation currently needs no rollback because it was not modified.

### Remaining uncertainties and upstream candidates

No Wan weights downloaded, full generation, quality comparison, donor throughput, or end-to-end model-loading test. Memory budget snapshots can race desktop allocations; activation/attention/VAE peaks, LoRA patching on compute and upstream cached assignments can exceed estimates. No automatic OOM retry or device transfer recovery is promised. An already-cached DisTorch assignment may bypass analysis on reload; ComfyUI's own reservation remains important. Reservations are conservative starting values and may need adjustment. Opt-in checks can reject loads that cache eviction would otherwise make possible.

This code cannot fix or prevent recurrence of NVIDIA's suspend/resume hang. Resolve driver/power-management diagnosis separately without triggering suspend during inference tests.

Potential small upstream fixes: respect core/model compute selection; validate donor input; use available rather than total donor capacity. Hardware-specific launcher/profile should be reviewed independently. Keep this log updated with future commit IDs, test results and model-specific observations before proposing changes to pollockjj.

## 2026-11-08 — installation and safe startup validation

Original clean installation backed up verbatim, including .git and untracked files, to `/home/claudebutnaru/AI/backups/ComfyUI-MultiGPU-original-20261108-114312`. Backup HEAD verified against original. Installed branch commit `bdd7de8dd0b6d4e2169c486914b34f8d7d64de94` via staged clone. No previously running ComfyUI server found.

All 10 regression tests pass from the installed copy. RAM-only safe-launch plan created. Startup test uses the same UUID environment and 2 GiB reservation, adding offline mode, browser auto-launch disabled, localhost-only port 8199. No prompts submitted or model weights loaded by the test. Startup/API results pending below.

### Completed validation

- RAM-only startup passed: ComfyUI 0.44.0 reports RTX 3060 at logical cuda:0; 26 MultiGPU nodes available via /object_info. Loader compute/device defaults inspected through API select cuda:0. /queue running and pending both empty.
- Optional-donor startup also passed: node choices include cpu, cuda:0 and cuda:1, while primary remains RTX 3060. GTX is visible for storage selection; this does not prove real model offloading or throughput.
- Both test servers stopped cleanly with SIGINT, exit code 0. Nothing left listening on test port 8199.
- No prompts, model downloads, generation or suspend/resume tests. No driver/display/power-setting changes. ComfyUI tracked working tree remains clean. No new NVIDIA assertion/Xid/blocked-sleep messages found in the recent kernel-log check.
- Startup initialized ordinary CUDA/runtime state (RTX memory observed around 808 MiB during first startup); this was not a full inference load. GTX stayed idle in the first RAM-only test. No throughput benchmark performed.
- Test artifacts in this chat's outputs: installation-record.json, installed-launch-plan.json, installed-launch-plan-donor.json, installed-test-results.json, installed-test-results-donor.json and comfy-startup-test*.log.
- Initial "not installed" notes above describe the earlier implementation stage. This section supersedes them: the branch is now installed, but servers are stopped. WanVideoWrapper is still absent; end-to-end Wan inference and cached-assignment behavior remain unverified.

Rollback: with ComfyUI stopped, move the current extension outside custom_nodes and copy the recorded original backup back to custom_nodes/ComfyUI-MultiGPU. Do not leave two copies under custom_nodes. Backup contains the original Git checkout; it has not been modified by testing.

For normal startup use the installed safe_launch.py with the ComfyUI environment's Python and --launch. The default selects only RTX 3060 plus RAM. Add the documented --donor option only when intentionally using GTX storage. Existing desktop launch shortcuts were not changed and may not activate this safety profile.

## 2026-11-08 — existing Wan model inventory

Requested next step: actual Wan offloading, starting small with existing weights only.

ComfyUI models directory contains only placeholders and YAML configs; no model symlinks and no extra_model_paths.yaml are present. No .safetensors/.gguf/.ckpt weights found in AI, Downloads or Projects. Hugging Face hub cache contains CLIP-ViT-B-32, stable-fast-3d and dinov2-large; no Wan repository. Targeted mounted-drive searches of CyData/models and CyData/model_lab found no Wan or safetensors/GGUF files within four levels.

Actual Wan test cannot proceed with the inspected model inventory. No downloads, prompts or generation started. An existing Wan diffusion-model path is needed; a generation test also needs matching text encoder and VAE. Do not substitute an unrelated model and report it as a Wan test. Broader filename search results are recorded below. Installed extension and original backup remain unchanged except this log.

Broader recursive filename search under home, /mnt and /media produced no matching Wan/UMT5 weights before being stopped after approximately one minute to avoid an unbounded scan of the old drive. This is not proof no weights exist elsewhere or under different filenames.

## 2026-11-08 — deeper search correction and real Wan test

Correction to the preceding inventory: deeper targeted search located complete Wan-AI/Wan2.1-T2V-1.3B-Diffusers weights inside the 2026-11-07 Codex workspace, under work/video-maker-data/models/hub. Generic filenames and snapshot symlinks caused earlier narrow searches to miss them. Never interpret the previous inventory as proof the machine lacked Wan weights. Also found unrelated Flux/Kronos/semantic-model weights. No Wan weights found by today's plocate index query; bounded broad scans timed out. An unmounted NTFS disk was not searched or mounted.

User authorized a 2.84 GB test-model download if needed; no download was needed or performed. Transformer source snapshot 0fad780a534b6463e45facd96134c9f345acfa5b has two complete FP32 shards (5,676,070,648 bytes combined), verified against safetensors header offsets. Five text-encoder shards and VAE also match header sizes. These checks establish structural completeness, not a cryptographic comparison against remote hashes.

Created separate models/diffusion_models/wan2.1_t2v_1.3B_local_fp32.safetensors (5,676,070,584 bytes), merging and renaming 825 keys. Every renamed key roundtrips exactly through the installed Diffusers convert_wan_transformer_to_diffusers function. No weight arithmetic, dtype change or original-cache modification. Conversion script and JSON report retained in this chat workspace.

Actual-model test uses installed UNETLoaderDisTorch2MultiGPU, FP32 source preservation, static model loading (--disable-dynamic-vram), 2 GiB reserve and UUID-verified primary. First target is 1 GiB offload to CPU; next target optional GTX. A single synthetic forward uses latent [1,16,1,4,4] and zero context [1,4,4096]; no text-encoder load, VAE decode or generated video. This exercises loading and block transfers rather than generation quality. Per-process timeout 180 seconds.

First test harness import failed because built-in extra nodes had not been initialized (TripleCLIPLoader missing). Corrected harness to initialize them before custom nodes, matching server startup order. No extension fix required for this harness error. Final results follow.

### Actual Wan offload results

Second harness correction: call the node's declared FUNCTION (override), rather than inherited load_unet directly. The inherited method does not accept MultiGPU fields. Both harness errors occurred before loading weights and are retained in wan-offload-harness-*.log. No production code was changed to resolve them.

Both final tests loaded real WAN21 weights through the installed DisTorch2 node and completed a single synthetic forward with finite output of shape [1,16,1,4,4]. Source FP32 preserved. RAM and GTX tests ran in separate short-lived processes; no server/queue was started and no video was generated.

| Target | RTX parameter bytes | Donor parameter bytes | Load seconds | Tiny forward seconds | RTX peak allocated bytes |
|---|---:|---:|---:|---:|---:|
| RAM | 4623660288 | 1052326912 | 2.481 | 1.226 | 4745357824 |
| GTX 1660 | 4623660288 | 1051208704 | 2.852 | 1.132 | 4745357824 |

GTX test also retained 1,118,208 parameter bytes on CPU (non-distributed parameters); this is expected from the existing loader's treatment of parameters outside its distributed modules. Approximately 4.31 GiB of weights on RTX and 0.98 GiB donor storage. Allocator peak is not total desktop VRAM; first-run observed nvidia-smi usage reached 5,568 MiB on RTX. Timing is a single synthetic smoke test, not a trustworthy donor performance comparison. No CPU-vs-GTX numerical equivalence or full-video quality comparison performed.

After tests, nvidia-smi returned to approximately 734 MiB RTX / 2 MiB GTX, temperatures 52 C / 32 C. Recent kernel-log check follows. Neither the original cache nor backup was changed. The separate converted 5.68 GB model remains available in ComfyUI. Future generation still requires compatible text-encoder/VAE loading and a bounded workflow; these were not attempted here. Static loading was tested; dynamic loading and cached-assignment memory changes remain unverified.

Artifacts: wan-local-conversion.json, wan-offload-cpu.json/.log and wan-offload-gtx.json/.log. Preparation and smoke-test scripts saved to this chat's outputs for reproducibility. This successful test supersedes the earlier missing-model blocker; no model download occurred despite conditional authorization.

Post-test check: no matching NVIDIA assertion/Xid/blocked-sleep messages in the last ten minutes of kernel logs. Core ComfyUI tracked tree is clean, and test port 8199 is not listening.

## 2026-11-08 — first bounded actual video with GTX offload

User authorized a small actual video using existing Wan models and RTX-primary/GTX-offload configuration. Planned output: 256x144, 17 frames at 16 fps (~1.06 seconds), 12 Euler/simple denoising steps, CFG 5, flow shift 8, seed 30601660. Prompt: "A small red sailboat gently rocking on a calm lake, soft morning light, slow ripples, locked-off camera." Negative: "blurry, distorted, flickering, text, watermark". Low-resolution smoke test, not a high-quality production preset.

Use existing local FP32 1.3B transformer through installed native ComfyUI DisTorch2 loader/sampler with 1 GiB GTX donor, 2 GiB RTX reserve and static loading. Existing Diffusers runtime handles local CPU UMT5 encoding and local RTX VAE decoding in separate processes; no WanVideoWrapper and no core ComfyUI changes. All stages force local/offline model loading. Existing source model files remain untouched. Native sampler process_out already unnormalizes Wan latents; decode must not apply that transform twice.

Resource monitor samples both GPUs and available RAM every two seconds. Abort above 10,240 MiB RTX or 5,120 MiB GTX usage, at >=82 C, or below 8 GiB available RAM; these are sampled limits, not a hard allocator cap. Per-stage time limits: encode 600s, sample 480s, decode 180s. Monitor stops only its own child process group if a limit or failure occurs. No suspend/driver/display/power changes. Final results follow.

### Completed actual-video result

PASS: produced wan-gtx-offload-small.mp4, verified by ffprobe as H.264/yuv420p, 256x144, 17 frames, 16 fps, 1.0625 seconds. All three monitored subprocesses exited successfully. No downloads. Native ComfyUI denoising ran on RTX 3060 with 1,051,208,704 bytes of transformer parameters on GTX 1660, 4,623,660,288 on RTX, and 1,118,208 on CPU. Sampler used the installed branch unchanged. No production-code fixes were required.

Text encoding: 148.0 seconds on CPU, using local FP32 UMT5 and 512-token padded prompt embeddings. Denoising: 19.244 seconds for 12 steps, finite latent shape [1,16,5,18,32]; RTX peak torch-allocated memory 4,903,590,400 bytes. Decode/export: 6.614 seconds, finite 17-frame tensor, RTX peak torch-allocated memory 1,317,302,272 bytes. These stage times are not overall wall time and are single-run measurements, not a benchmark against RAM-only operation.

Sampled monitoring summary: {"sampled_peak_by_nvidia_smi_index": {"0": {"memory_mib": 1149, "temperature_c": 36}, "1": {"memory_mib": 5945, "temperature_c": 65}}, "minimum_available_ram_gib": 48.3006706237793, "limits_triggered": false}.

Visual inspection of first/every-fourth/final frames shows a small sailboat scene with frame-to-frame variation. The tiny 12-step preview is soft, saturated and has limited motion; it is not evidence of production-quality Wan output or equivalence with an unmodified reference. Mean absolute frame change is 3.514/255 and pixel standard deviation is 61.184, useful sanity checks rather than quality scores. No full-resolution, long-duration, repeated-load, dynamic-loader or numerical baseline comparison performed.

After completion both GPU processes released resources: approximately 765 MiB RTX / 2 MiB GTX, 49 C / 33 C at final check. No matching NVIDIA assertion/Xid/blocked-sleep messages in recent kernel logs. Core ComfyUI tracked tree stays clean. No server was started. No driver, display, suspend/resume or power settings changed; prior suspend/resume issue remains unresolved.

Reproducibility artifacts in this chat outputs: video_settings.json, video_text_decode.py, video_sampler.py and run_small_video.py; reports wan-video-text.json, wan-video-sampling.json, wan-video-decode.json, wan-video-run.json, wan-video-resource-summary.json; raw stage logs and monitor JSONL. Embeddings, sampled latents and PNG frames remain under work/ for follow-up. Original local model cache and backed-up extension remain intact. The video uses Diffusers only for text/VAE components and native ComfyUI plus DisTorch2 for actual denoising/offload; this does not validate WanVideoWrapper, which is absent.

## 2026-11-08 — matched GTX versus RAM video comparison

Authorized settings: same sailboat prompt/negative, seed 30601660, FP32 weights and conditioning, CFG 5, shift 8, Euler/simple; increased to 512x288, 33 frames at 16 fps, 20 steps. Each run offloads 1 GiB to either GTX or RAM. Reuse the exact cached prompt embeddings from the first successful video; no text-encoder rerun or downloads required. Separate work/quality-gtx and work/quality-ram folders and outputs/quality-* artifacts preserve the first video.

Same sampled monitoring limits and stage timeouts as first video. Initial comparison launch accidentally invoked the old harness and began unnecessary re-encoding; stopped that child before completion. A second setup launch omitted donor environment selection and was stopped before the actual comparison. Both were harness invocation errors, not extension failures. The top-level wan-video-run.json/encode log from the earlier test were overwritten by the aborted old-harness launch; original MP4, model files, sampling/decode reports and original log results are preserved. Only correctly configured completed comparison runs are used below.

### Matched comparison results

Both complete 512x288 / 33-frame / 20-step videos passed and all child processes exited cleanly. GTX sampling: 163.130 seconds; RAM: 147.726 seconds (RAM 9.44% lower sampling time in this single sequential pair). Latent tensors were exactly equal (max absolute difference 0), and first decoded-frame pixels were equal. GTX storage: 1,051,208,704 parameter bytes; RAM storage: 1,052,326,912 bytes; RTX placement identical.

Sampled GPU peaks (MiB, C): GTX run RTX 7173/77 and GTX 1149/40; RAM run RTX 6990/80 and GTX idle 2/38. Minimum available RAM 47.91 GiB / 48.25 GiB. No stop threshold reached. RAM run occurred second with warmer GPU and caches; no repeated benchmark or broad speed claim. Favor RAM as default for this workload; GTX remains optional tested storage. Videos and reports live under outputs/quality-gtx and outputs/quality-ram; comparison in outputs/quality-comparison.json.

Larger preview visually shows a clearer red sailboat with water detail than the first tiny video. This is still a short low-step test, not a production-quality guarantee. No driver/display/power changes or new model downloads. Subsequent Resources UI customization is a separate codebase and must not be bundled into MultiGPU code.

## 2026-11-08 — controlled quality refinement, 30 steps

User requested improved video quality with RTX 3060 primary. Preserve the successful 20-step baseline. New run: same local FP32 Wan 2.1 T2V 1.3B, sailboat prompt and cached embeddings, seed 30601660, 512x288, 33 frames at 16 fps, CFG 5, flow shift 8, Euler/simple. Only denoising steps increase 20 to 30. RAM offloading (1 GiB) chosen from the earlier matched pair; GTX remains optional and is not exposed to this run. UUID mapping selects RTX 3060 explicitly. No model downloads, driver/display/power changes or upstream modifications.

Use existing two-second resource sampling, RTX 10,240 MiB cap, GTX 5,120 MiB cap, >=82 C temperature stop, minimum 8 GiB available RAM and bounded sampling/decode stages. These sampled checks are not a hard allocator limit. Results will be compared visually and numerically with the 20-step baseline; more steps are not assumed to guarantee better quality.

### Completed 30-step refinement

PASS: RTX-primary/RAM-offload run completed 30 Euler/simple steps in 218.269 seconds (20-step baseline 147.726s). Decode/export 13.977s. MP4 verified at 512x288, 33 frames, 2.0625s. Finite sampled/decoded tensors; RTX peak torch allocation 5,787,621,888 bytes. Sampled device peaks: RTX 6971 MiB / 79 C; GTX 2 MiB / 37 C. Minimum available RAM 47.52 GiB. No monitoring threshold reached; both stage processes exited successfully. No new matching NVIDIA assertion/Xid/blocked-sleep messages found in recent kernel check.

Saved outputs/quality-30steps/wan-quality-30steps.mp4 and 20-vs-30-steps.jpg. Visual comparison shows more defined hull/rigging and changed water detail, but imperfections remain; extra steps alone do not establish a large or objective quality gain. Pixel difference from baseline is recorded only as evidence of change, not a quality score. Baseline preserved. RTX returned to desktop-level memory after process exit. No code changes, model downloads, driver/display/power changes.

## 2026-11-08 — developer example workflow audit

User requested examining the developer's examples to understand stronger demo results. Started with pollockjj's bundled examples in the base fork and cross-checked the official ComfyUI Wan2.2 guide and kijai/WanVideoWrapper. No exact demo link was supplied, so findings describe inspected workflows, not a proven causal explanation of an unspecified showcase clip.

- wan2_2 distorch2 double_unet no_cpu.json: high/low noise Wan2.2 14B FP8 scaled models, 1280x704, 57 frames, 20-step Euler/simple schedule split at step 10, CFG 3.5, shift 8. Both model nodes request 9 GiB virtual VRAM from cuda:1; VAE/text encoder also assigned there. The filename/note is not a reliable substitute for inspecting actual nodes and wiring.
- wan2_2 t2v lightx2v lora distorch2.json: same 14B high/low model family, matching high/low LightX2V/Seko LoRAs, 768x768, 81 frames, 8-step Euler/simple schedule split at 4, CFG 1. One model donates to GPU and the other to CPU in this saved graph. These accelerated model/LoRA settings cannot be blindly applied to our plain Wan2.1 1.3B model.
- ComfyUI-WanVideoWrapper wanvideo_T2V.json: 14B Wan2.1 model, 832x480x81, UniPC sampler, 25 steps, CFG 6, shift 5; Enhance-A-Video, TeaCache/block-swap/compile configuration appears in the graph. Cache/compile/swap settings concern execution and require compatibility checks rather than being universal quality knobs.
- ComfyUI-WanVideoWrapper wanvideo_1_3B control_lora.json: 1.3B model with a tile control LoRA and an existing input video, not plain text-only generation; UniPC 30 steps/CFG 6/shift 5. Video conditioning and control materially change the task.
- Wan2.2 I2V A14B GGUF example uses image conditioning, high/low noise quantized 14B models and matching LoRAs. This is not equivalent to our unconditioned small T2V baseline.

Major differences: model family/capacity, conditioning, resolution/frame count, and acceleration trained for a particular model. Current offload work enables placement but does not upgrade model capability. No standalone post-upscaler was present in the two reviewed native T2V graphs; do not assume all visible quality came from post-processing.

Hardware adaptation: never copy example cuda:0/cuda:1 identities or 9/13 GiB donor settings. The GTX has 6 GiB total and cannot supply those requests alone. Preserve UUID-verified RTX primary, desktop reserve and RAM fallback. Two 14B experts and their LoRAs require a separate memory/download plan; no such model download is authorized by this audit.

Official ComfyUI guide says Wan2.2 TI2V 5B should fit 8 GB VRAM with native offloading, making it a candidate intermediate upgrade, not a local guarantee or a match for the demonstrated 14B workflow. Its Wan2.2 VAE differs from the VAE used by the 14B examples. Link: https://docs.comfy.org/tutorials/video/wan/wan2_2 . Official Wan reference-script memory figures differ from optimized ComfyUI workflows; do not mix their requirements.

Audit details and source links saved in outputs/developer-workflow-audit.json. Original examples: https://github.com/pollockjj/ComfyUI-MultiGPU/tree/main/example_workflows . Related wrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper . Next quality work should choose a model/conditioning strategy from these verified differences before increasing workload further.

## 2026-11-08 — first local AI production assistant

Built a separate local Production Studio around the verified ComfyUI/Wan pipeline, following the developer-workflow audit rather than copying incompatible 14B/LoRA settings into the 1.3B test. Existing local Qwen 3 4B instruct plans one to three editable shots on CPU; verified it unloads after planning. Native ComfyUI/MultiGPU handles Wan denoising, with existing local Diffusers text/VAE workers separated by process. RTX selected by UUID; RAM default and optional GTX storage. No model downloads, new driver/power settings or production ComfyUI/extension code changes.

Working UI includes planning, editable shots, bounded profiles, landscape/portrait/square, reproducible seeds, one active job, scoped stop controls, MP4 assembly, saved manifests and history. Loopback-only service on 127.0.0.1:8765; local Production Studio menu launcher. Local source project lives separately in this chat workspace, with source/help ZIP and local-config backup in outputs. No new public code repository was created for this app.

End-to-end browser test planned two shots, retained one, ran a real new 256x144/17-frame/12-step RAM-offload generation, exported and played its MP4. Total job elapsed 59.4s. Sampled peaks: RTX 6008 MiB / 66 C; GTX 2 MiB / 33 C; minimum available RAM 47.28 GiB. Finite tensors and browser-loaded video metadata verified; finished job survived service restart. Result is a rough proof, not a production-quality showcase.

15 automated checks passed, including validation, UUID/resource limits, restart records, overlap prevention, safe local HTTP/media ranges, protected paths, real FFmpeg two-clip fixture assembly and cancellation of an owned sleeping subprocess. JavaScript executed successfully in browser; no Node installation needed. Existing source examples and official model guide are linked in app/help. Silent clips, hard cuts and basic T2V continuity remain first-version limits; stronger models need separate review. Details in the app ENGINEERING_LOG.md and outputs/production-studio-verification.json.

## 2026-11-08 — resume quality work after desktop repair

Confirmed compositor ignores GTX and uses RTX after reboot; user reports tearing stopped. Starting isolated UniPC/simple versus preserved Euler/simple 30-step comparison: same Wan2.1 1.3B FP32 checkpoint, prompt embeddings, seed 30601660, shift 8, CFG 5, 512x288, 33 frames at 16fps. RAM donor selected based on prior matched test. UUID-pinned RTX primary; UUID-based 2s monitoring, 10240MiB primary/5120MiB donor budgets, 82C stop, 8GiB RAM floor, 480s sample/180s decode limits. No downloads or production code changes. Baselines preserved. UniPC implementation may differ from Wan reference solver; quality pending comparison.

### UniPC result and bounded resolution preview

PASS: UniPC/simple 30-step comparison completed, sampling 210.916s, decode/export 19.652s (stage totals include imports/load). 512x288, 33 frames, 2.0625s, finite latents/decoded frames. Sampled RTX peak6813MiB/78C; GTX2MiB/35C; available RAM floor48.37GiB. Matching frames show cleaner mast/rigging and altered background/water; subjective preference for this seed only. No monitoring limit reached. Baseline preserved in quality-30steps; comparison and exact worker/settings snapshots in outputs/quality-unipc. Reference: https://github.com/Wan-Video/Wan2.1/blob/main/generate.py ; native ComfyUI UniPC is not asserted numerically equivalent.

Next isolated preview: UniPC/simple, 768x432, 17 frames (1.0625s), 30 steps, same prompt/seed/CFG5/shift8, RAM offload. Higher pixel count and shorter duration both change latent noise/scene, so this is a feasibility/detail preview rather than a single-variable quality comparison. Same UUID monitoring and resource stops. No downloads.

### Completed quality session

PASS higher-resolution preview: 768x432,17 frames,16fps,1.0625s,30 UniPC/simple steps, sampling 290.591s; finite samples and decoded frames. Sampled RTX peak {'memory_mib': 7101, 'temperature_c': 80}; RAM floor 49.74GiB. No safety threshold reached. More water/sail detail visible but scene/geometry changed; no controlled proof that resolution alone improves output. Keep experimental, no new larger default. Exact reports, workers/settings, MP4 and frame saved in outputs/quality-higher-resolution.

Production Studio Detail sampler now UniPC on local quality/unipc-detail commit71332bd; Quick/Balanced unchanged. 15 existing tests passed; original source backed up. Loopback service restarted and /api/status reports ready with no active jobs. No GPU generation is left running. Models, ComfyUI core and extension code untouched; no new downloads, drivers, suspend or display changes during quality work. Suspend/resume issue remains unresolved and untested.

## 2026-11-08 — requested higher resolution and approximately ten seconds

User authorized extending quality/duration. Before increasing workload, run isolated BF16-UNet compatibility/quality check at same 768x432x17,30 UniPC/simple,CFG5,shift8,seed30601660; RTX UUID pinned, CPU donor, existing FP32 checkpoint cast by native ComfyUI --bf16-unet. Original checkpoint and tested FP32 profiles preserved; VAE remains FP32. BF16 may change output and must pass finite tensor and visual checks. Same 2GiB desktop reserve, 82C stop, >=8GiB RAM and stage bounds. No new weights/downloads/driver/display changes. Duration target is approximately161 frames at16fps, not a slowed or looped short clip. Feasibility pending.

### BF16 compatibility check passed

768x432x17,30 UniPC:66.110s sampling versus290.591s prior FP32, finite samples/decode; parameters confirmed BF16 with FP32 small components. Frames visually closely match preserved FP32 sailboat at this seed; equivalence not claimed. Resource peaks by UUID:{"GPU-888a542e-afbd-c60e-8458-28cadb4ce8ce": {"memory_mib": 2, "temperature_c": 39}, "GPU-d5ac3456-9f36-f11c-aee1-406ddc9332a6": {"memory_mib": 3455, "temperature_c": 76}}. Starting separate 832x480x161,3-step memory/thermal/decoder probe; low steps are solely a feasibility check, not a deliverable quality clip. Worker-local callback synchronizes after steps and idles at >=76C until <=68C (120s bound), independent outer82C stop remains. No system fan/clock/power settings changed. Probe sample limit600s/decode300s.

### Ten-second probe thermal stop

Original 832x480x161 three-step probe stopped by outer monitor at82C after140.36s; RTX memory6626MiB at stop, ~6.65GiB-class sampled use, RAM safe. First UniPC step completed in81.35s including4.06s cooldown from80C to68C. Following long step reheated before next callback; step-only cooling insufficient. No successful latents or finished clip from aborted probe; no driver errors claimed. Preparing separate repeat with synchronous GPU temperature checks every two transformer blocks and worker-only idle at75C to67C; outer82C and memory limits unchanged. No clocks/fans/power/driver changes. Originals preserved.

User steering: use both cards for both finished versions. Prepared batch switched to TEST_OFFLOAD=gtx; UUID order RTX first/GTX second and donor model-name assertion. GTX receives model weight storage, RTX performs UNet/attention and VAE; not parallel denoising on GTX. CPU residual/fallback remains supported. Current RAM thermal probe will finish; separate BF16 GTX3-step small compatibility check required before finished jobs. Plan updated; safety budgets unchanged.

### Full-shape probe passed with intra-block cooling

832x480x161,3 UniPC steps completed in398.052s including162.542s worker-only cooldown. Finite sampled tensors and full161-frame FP32 VAE decode/export (158.999s). Sampled resource peaks:{"GPU-888a542e-afbd-c60e-8458-28cadb4ce8ce": {"memory_mib": 2, "temperature_c": 42}, "GPU-d5ac3456-9f36-f11c-aee1-406ddc9332a6": {"memory_mib": 6817, "temperature_c": 81}}. This low-step feasibility clip is not the requested finished quality result. Decoder became warm; finished workers now also check temperature between VAE temporal chunks at75C and idle to67C. Original82C external cutoff and memory reserves retained. Starting small BF16+GTX compatibility check before finished jobs. User explicitly asked to try both continuous/three-shot versions and use both cards.

### BF16 GTX storage compatibility passed

768x432x17,3 UniPC steps, finite sampling/decode with RTX parameter bytes1773849216,GTX1063784960,CPU559104. Sampling9.487s. Both GPUs selected by UUID; GTX has no BF16 denoising workload, only weight storage. Proceeding with user-requested both-card finished versions, three-shot sequence first. Controller and exact plans saved in outputs/quality-ten-seconds; no concurrent GPU jobs.

2026-11-08T14:45:51.744995-06:00 Started both requested versions sequentially: three53-frame shots followed by one161-frame continuous shot; all832x480,30 UniPC/simple steps,BF16 UNet/FP32 VAE,GTX1660 donor,UUID-pinned RTX. Intra-block cooling75C to67C; outer82C cutoff retained. No new models or system graphics/power changes.

2026-11-08T14:46:20.376391-06:00 START quality-ten-second-shot-01

Cooling timing refinement for NOT YET STARTED shot02,shot03,continuous workers only: poll cooldown temperature every0.25s instead of2s, retaining75C trigger,67C resume target,120s cooldown bound,82C external cutoff and2s outer monitoring. Current shot01 worker/source unchanged. This avoids continuing to idle after the same cooling target is reached; no change to model calculation or system hardware policy. Exact executed source snapshots retained per run.

2026-11-08T15:02:22.506754-06:00 PASS quality-ten-second-shot-01: {"sampled_peaks": {"GPU-888a542e-afbd-c60e-8458-28cadb4ce8ce": {"memory_mib": 1178, "temperature_c": 48}, "GPU-d5ac3456-9f36-f11c-aee1-406ddc9332a6": {"memory_mib": 5302, "temperature_c": 76}}, "sampling_seconds": 844.307639572, "cooling_seconds": 434.559149106994}

2026-11-08T15:03:07.457395-06:00 START quality-ten-second-shot-02

## 2026-11-08 — user requested researched acceleration and GitHub publication

Preserved completed shot01 as baseline; remaining batch held at explicit PERFORMANCE_GATE before shot02 GPU work. New fork branch performance/wan-3060-safe-speed, based on hardware branch29818e7. Native ComfyUI already prioritizes Flash SDPA; no need to install it again. Existing Comfy Kitchen INT8 kernel microbenchmark (RTX SM86,BF16,batch2,12 heads,head128,16384tokens,3 repeats,including quantization)133.043ms SDPA versus51.817ms INT8 (~2.57x kernel-only), finite results, relative RMS error~0.0167 on random tensors. Not whole-video speed or quality. SOL tau0.5 was faster but random-tensor RMS discrepancy~0.69; not selected for quality presets. No new packages installed. Sage2 source build would confront local CUDA toolkit12.4 versus Torch13.2; prefer existing native kernels. Primary-source references saved in performance documentation.

Implemented opt-in BF16 safe-launch flag and native Wan model-patcher helper (INT8 attention and optional EasyCache; defaults preserve PyTorch/no-cache, explicit unsupported-component fallback reports, no global monkeypatch). 17 CPU-only tests passed. Starting matched three-way real Wan tests at768x432x17,30UniPC,CFG5,shift8,same prompts/seed/BF16/both-GPU placement/cooling0.25s. Compare native SDPA, native INT8, INT8+EasyCache0.10 (15–85% window). No native-cache benefit assumed before measurement and visual review. Remaining long jobs stay gated.

Matched short result: native INT8 without cache98.133s versus native SDPA102.535s sampling, only~4.3% reduction in this full workload despite2.57x attention microkernel result. INT8 cooling6.70s; SDPA0s. Whole-model layers/offload and cooling dominate. Do not advertise2.57x whole-video gain. Preparing separate nativeINT8+0.25GiB donor run, with everything else held fixed, to reduce GTX PCIe weight transfer overhead while retaining both-card storage and full desktop reserve. No default change before measurement.

Transfer test passed: nativeINT8 donor0.25GiB sampled78.744s versus98.133s at1GiB (same768x432x17,30steps), ~19.8% shorter sampling; actual GTX parameter bytes256514048, RTX2581120128, CPU559104. Both cards used, reservations retained. NativeINT8+EasyCache0.10 at1GiB donor sampled54.281s, skipping12/30 steps. Visual contact sheet shows meaningful boat/sail geometry/composition changes with cache; keep approximate cache opt-in and do not claim quality equivalence. Testing compute-stream thermal fencing against whole-device fencing next, with exact sameINT8/no-cache/0.25GiB donor configuration; global drain still occurs before cooling and final report.

### Controlled performance/quality findings completed

All six768x432x17 BF16 both-card30-step cases passed finite sample/decode and MP4 export. Sampling seconds: {"performance-sdpa": 102.535, "performance-int8": 98.133, "performance-int8-small-donor": 78.744, "performance-int8-cache": 54.281, "performance-stream-fence": 88.829, "performance-balanced": 53.031}. Exact stage totals, placement, resource peaks and cache counts retained in benchmark report. Reducing nativeINT8 donor1GiB to0.25GiB preserved sampled latents bit-for-bit and cut98.133s to78.744s. Stream-fence variant likewise bit-identical, but88.829s versus78.744s; cooling15.928s versus6.431s, no independent speed gain established. EasyCache0.10 skipped12/30 steps and visibly changed geometry; combined0.05+0.25GiB+reusablepacer skipped9/30,53.031s, still approximate and not a quality-equivalent default.

Quality recipe therefore keeps cache OFF, uses nativeINT8 attention and0.25GiB GTX weight storage, preserving RTX primary and headroom. Optional cache stays explicit. Reusable ThermalPacer passed CPU safety tests and actual GPU sample/decode; all24 extension CPU tests passed. Before remaining full-resolution jobs, install only tested helper/launcher/recipe/docs with original extension backup. User explicitly authorized publishing to fork; feature branch performance/wan-3060-safe-speed created from29818e7,main unchanged.

Installed tested opt-in performance modules,launcher,recipe,benchmark/help into existing extension after full source backup at /home/claudebutnaru/AI/backups/ComfyUI-MultiGPU-before-performance-20261108-152637. Core/model files unchanged. New modules have no automatic import-time GPU patches; existing defaults stay unchanged. Remaining owned video workers will explicitly select nativeINT8,0.25GiB donor,cacheOFF and bounded thermal pacing. Public benchmark includes both approximate cache tests, with visual tradeoff stated.

User explicitly approved temporary130W RTX power-cap experiment with restoration to170W. Queried target UUID/RTX3060 12GB: current/default170W,allowed100–170W. Prepared dry-run-first bounded administrator helper; only NvidiaSMI runs privileged, model workers remain normal user. Original cap restored in finally/on interrupt/after900s or stop-file, with readback and retries. All28 CPU tests passed, including restoration on normal completion and worker/user interruption. Current170W full832x480x53 confirmation passed: sampling421.242s versus preserved844.308s baseline, cooling157.647s, finite decode93.978s; visual frames checked, no matching fresh NVIDIA Xid/assertion found. Administrator authentication pending before130W run; do not treat approval alone as applied cap.

Full-resolution software confirmation at170W: 832x480x53,30steps,cacheOFF, same prompt/seed: baseline844.308s sampling/961.032s load+sample+decode stages; accelerated421.242s/528.350s (~2.00x sampling,~1.82x stage wall). RTX sampled peak6145MiB/76C, GTX354MiB/46C. Original baseline preserved. New frame geometry visually similar with INT8 differences, not pixel-equivalent. Public exact report added. Temporary130W cap successfully authenticated/applied by UUID; matched run in progress and power limit logged on every resource sample. Automatic restoration helper remains active; no recommendation before completed comparison.

Published and read-back verified feature branch performance/wan-3060-safe-speed commit91dc469fa272c4456df85214468ced532e3d2b49; local and installed branch aligned after byte-for-byte verification, main/upstream unchanged. Matched130W experiment passed:356.613s sampling (170W421.242s), cooling85.613s vs157.647s; stage wall452.768s vs528.350s. All53 PNG frames True, sampled latents bit-identical. Observed cap130W throughout; helper restored170W and live readback confirmed. User separately approved130W only for queued renders, restore170W after completion. Bounded helper maximum expanded explicitly to7200s, default900s unchanged;30 CPU tests passed. Added nonprivileged power watcher to signal restoration on batch complete/failure; completed/failed cleanup fixtures passed without GPU access. Original baseline shot01 archived; validated accelerated shot01 replaces it so final sequence uses nativeINT8/0.25GiB donor/cacheOFF consistently.

User explicitly approved adopting130W only for the queued videos, with original170W restored after completion. Prepared7200s administrator helper session with independent nonprivileged power trace/cleanup watcher; batch remains gated until authenticated cap readback. All30 CPU tests and watcher complete/failed cleanup fixtures passed. Standalone power experiment compared exact53 PNGs and MP4 bytes: both identical between170W/130W, in addition to bit-identical sampled latents. Actual trial restored170W before the new session.

Approved remaining-render130W session authenticated and live-readback verified; independent watcher running and will signal170W restoration on batch complete/failure (7200s helper deadline). Released performance gate. Remaining shots/continuous use installed nativeINT8,0.25GiB GTX weight storage,cacheOFF,UUID-verified RTX primary and unchangedVRAM/RAM/82C stops. Models remain unprivileged; no unrelated processes targeted.

2026-11-08T16:04:55.841976-06:00 PASS quality-ten-second-shot-02: {"sampled_peaks": {"GPU-888a542e-afbd-c60e-8458-28cadb4ce8ce": {"memory_mib": 354, "temperature_c": 45}, "GPU-d5ac3456-9f36-f11c-aee1-406ddc9332a6": {"memory_mib": 6128, "temperature_c": 75}}, "sampling_seconds": 295.9204813509996, "cooling_seconds": 25.758889694001482}

2026-11-08T16:05:38.725963-06:00 START quality-ten-second-shot-03

2026-11-08T16:13:36.816403-06:00 PASS quality-ten-second-shot-03: {"sampled_peaks": {"GPU-888a542e-afbd-c60e-8458-28cadb4ce8ce": {"memory_mib": 354, "temperature_c": 48}, "GPU-d5ac3456-9f36-f11c-aee1-406ddc9332a6": {"memory_mib": 6109, "temperature_c": 75}}, "sampling_seconds": 376.37477671600027, "cooling_seconds": 103.30481138900268}

2026-11-08T16:13:36.921950-06:00 PASS three-shot sequence: native 832x480,159 frames at16fps,9.9375s. Three independently generated53-frame shots, hard cuts; no slow-down, repetition or interpolation. Identity continuity not guaranteed.

2026-11-08T16:14:30.052910-06:00 START quality-ten-second-continuous

2026-11-08T16:56:00.313686-06:00 PASS quality-ten-second-continuous: {"sampled_peaks": {"GPU-888a542e-afbd-c60e-8458-28cadb4ce8ce": {"memory_mib": 354, "temperature_c": 47}, "GPU-d5ac3456-9f36-f11c-aee1-406ddc9332a6": {"memory_mib": 7630, "temperature_c": 75}}, "sampling_seconds": 2196.454462615, "cooling_seconds": 920.8150365709853}

2026-11-08T16:56:00.359117-06:00 Both native-resolution versions complete. All workers exited. GPU reserves and cutoff unchanged; visual QA still required.


## Completed ten-second comparison — 2026-11-08

Both native 832×480, 16 fps videos completed with finite sampling and decoding: continuous 161 frames (10.062 s), and three independent 53-frame shots (159 frames, 9.938 s). No upscaling or frame interpolation was used. Sampled first/middle/last frames show a consistent boat and horizon in the continuous clip; the independent shots vary in boat design and lighting. This frame inspection does not establish perfect temporal consistency.

The continuous run sampled in 2,196.45 s, decoded in 277.70 s, and took 2,488.32 s across measured stages (41.47 minutes). Cooling accounted for 920.82 s of sampling, so sustained cooling remains a major bottleneck. Peak RTX allocation reported by monitoring was 7,630 MiB at 75°C; GTX peak was 354 MiB at 47°C, with at least 50.84 GiB RAM available. The RTX remained compute/display primary; the GTX stored offloaded weights rather than running parallel denoising. Native INT8 attention, BF16 transformer, FP32 VAE and 0.25 GiB donor settings were used, with approximate step caching disabled.

The explicitly approved temporary 130 W limit was observed throughout the monitored batch. After completion, the administrator helper restored the original 170 W RTX limit; independent NVIDIA readback confirmed it. All owned render workers, the watcher and power helper exited successfully. No matching current-boot NVIDIA Xid/assertion or blocked-task entries were found in the journal check. Suspend/resume was not exercised or fixed. Existing originals remain preserved.

See [the completed measurement report](https://github.com/Cyclotron555/ComfyUI-MultiGPU/blob/performance/wan-3060-safe-speed/benchmarks/results/rtx3060-ten-seconds.json). The prior 30 CPU-only tests passed; this final update changes documentation and recorded results only. The small 1.3B model and extended sequence length still limit fidelity and temporal reliability; these results are not a general performance guarantee.

## Cooling and walking-human phase — 20261108-170316

User requested faster videos without quality loss and a human walking. Backed up installed extension to /home/claudebutnaru/AI/backups/ComfyUI-MultiGPU-before-cooling-20261108-170316. Baseline desktop RTX47C/170W, GTX40C. NVIDIA reports SW/HW thermal slowdown counters0; software guard pauses at75C, well below reported83C target/93C maximum operating temperature. Keep conservative guard unchanged. NVML read-only temperature query measured8.5us versus38ms per nvidia-smi process (~24s theoretical savings across630 checks, not an end-to-end speed claim). XWayland fan settings are unavailable; direct NVML reports two RTX fans, both automatic policy0, speed35%, allowed30–100%. Investigate a bounded75% fan experiment at130W through documented NVML API, restoring automatic control and170W. No display/driver changes.

2026-11-08T17:08:57.332763-06:00 Cooling/walking controller stopped: KeyboardInterrupt()

Pre-render setup adjustment: automatic fan-stop at idle held RTX51–52C, so a50C starting target was unsuitable. No model worker had started. Interrupted controller normally, verified helper restored automatic fans and170W, preserved this idle-only attempt separately, and use the previous validated55C between-run start ceiling. The75/67C pacing and82C worker cutoff remain unchanged.

2026-11-08T17:09:28.487002-06:00 START walking-baseline

Cooling code validation:41 CPU-only tests pass, including UUID mismatch/fork/query-error handling, no driver initialization at import, no fan writes during planning, refusing existing manual policy or lowering automatic target, restoring both fans after partial setter failure,80C stop, and integrated interrupt restoring fans+170W and signaling controller abort. Added explicit NVML backend (SMI remains default), read-only telemetry benchmark, opt-in temporary fan flags and walking recipe. Actual100-query read-only probe: NVML1.81us/query vsSMI37.31ms/query. End-to-end benefit is not yet established. Existing encode weights used locally on CPU;145.05s to encode walking prompt, no download.

2026-11-08T17:16:18.525682-06:00 PASS walking-baseline: {"sample_seconds": 318.9209313319989, "cooling_seconds": 47.54633017899687, "decode_seconds": 77.63631539400012, "maximum_guard_temperature_c": 75}

2026-11-08T17:16:53.646938-06:00 START walking-cooled

User steering: render the final walking clip in Japanese cartoon/anime style. Keep the already-running matched photorealistic53-frame comparison unchanged for valid cooling/quality evidence. Updated only the pending161-frame final prompt to hand-drawn2D anime, cel shading, painted park background, one blue-jacket adult walking side-on. Added fresh CPU-only/offline text encoding before the final sampler; no stale photorealistic embeddings, new downloads, model change, reduced steps, upscaling or interpolation. Updated public recipe draft; will publish with completed results.

Interim75percent trial still required frequent75C-to67C cooling. Do not assume faster temperature reads improve end-to-end speed: removing CPU query gaps increases sustained GPU load and heat. Prepare a90percent temporary-fan trial, same130W, sameguard and exactsame53-frame walking comparison. Preserve75percent trial including any negative result; reuse original baseline, explicitly note sequential/ambient/foreground confounding. The pending161-frame final is now Japanese2D anime walking; fresh CPU encoding precedes sampling.

2026-11-08T17:24:23.690342-06:00 PASS walking-cooled: {"sample_seconds": 355.99309719899975, "cooling_seconds": 97.79741685300178, "decode_seconds": 80.07992394800021, "maximum_guard_temperature_c": 75}

2026-11-08T17:24:23.765638-06:00 Matched walking quality check: {"latents_file_identical": true, "frames_compared": 53, "all_png_bytes_identical": true, "mp4_bytes_identical": true}

2026-11-08T17:24:23.766188-06:00 Cooling/walking controller stopped: RuntimeError('No matched sampling speed improvement; skipping long render for review')

Completed75percent trial: baseline318.9209s sampling/47.5463s cooling;75percent+NVML355.9931s sampling/97.7974s cooling (11.62% slower). Latents, all53 PNG bytes and MP4 bytes exactly identical. Controller correctly rejected a slower candidate and skipped its long render. Verified helper exited0 after restoring automatic fans and170W. Initiated90percent follow-up with same130W, originalguard and reused baseline. Negative result preserved.

2026-11-08T17:25:06.891932-06:00 Reused completed53-frame walking baseline318.92s, rather than repeating an identical expensive render;90percent fan candidate uses same prompt/seed/settings. Original75percent trial preserved separately.

2026-11-08T17:25:15.499607-06:00 START walking-cooled-90

2026-11-08T17:31:57.549149-06:00 PASS walking-cooled-90: {"sample_seconds": 315.6574740830001, "cooling_seconds": 57.27650631000688, "decode_seconds": 72.71613912000066, "maximum_guard_temperature_c": 75}

2026-11-08T17:31:57.625574-06:00 Matched walking quality check: {"latents_file_identical": true, "frames_compared": 53, "all_png_bytes_identical": true, "mp4_bytes_identical": true}

2026-11-08T17:32:36.749672-06:00 START walking-anime-ten-seconds

90percent/75C trial completed315.6575s sampling57.2765s cooling versus baseline318.9209/47.5463. Matched latents, all53 PNGs and MP4 remain identical. Nominal1.02% sampling-time reduction is too small for a reliable performance claim. Prepare a bounded77C pause/72C resume pilot, preserving82C worker cutoff and stricter80C privileged-helper abort. Manufacturer reports83C target,93C max-operating; no hardware temperature/driver limit is changed. Final anime sampler explicitly waits for this pilot and exact-output validation; only local CPU prompt encoding overlaps, never two GPU render workers. Pilot keeps53 frames,30steps,allmathsettings,seed/prompt identical. VAE remains at75C/67C. Record CPU-encoding overlap as timing confound.

2026-11-08T17:33:59.228446-06:00 Cooling/walking controller stopped: RuntimeError('Cooling experiment stopped: []')

Sequencing correction: the previous final had already begun loading its sampler when the planned pilot gate was written, so stopped that owned controller before launching any pilot. No simultaneous GPU render workers were started. Its completed anime CPU embeddings are preserved and reused (freshly encoded from anime prompt). The90percent helper exited0 and restored automatic fans+170W. A new isolated77C pilot controller will run53-frame validation then the161-frame anime render sequentially; final sampler waits on explicit validation gate. Skip redundant CPU re-encoding and record every attempt.

2026-11-08T17:34:55.264090-06:00 Reused completed53-frame walking baseline318.92s, rather than repeating an identical expensive render;90percent fan candidate uses same prompt/seed/settings. Original75percent trial preserved separately.

2026-11-08T17:35:03.882744-06:00 START walking-cooled-77

User explicitly asked to see anime, not the already-completed realistic walking clip. Prepared a clearly labeled short style preview:512x288,33frames16fps,20steps, same anime prompt and existing model/embeddings. The full10s remains832x480,161frames,30steps. Added an explicit pre-Torch gate to the unstarted10s sampler so the style preview runs alone after the77C validation. No concurrent GPU generation. Audio had not yet been created or mixed; a separate light-foley/ambience pass was planned after motion validation.

2026-11-08T17:41:17.919069-06:00 PASS walking-cooled-77: {"sample_seconds": 280.030340936999, "cooling_seconds": 19.25869421099742, "decode_seconds": 82.57794253800057, "maximum_guard_temperature_c": 77}

2026-11-08T17:41:17.963144-06:00 Matched walking quality check: {"latents_file_identical": true, "frames_compared": 53, "all_png_bytes_identical": true, "mp4_bytes_identical": true}

2026-11-08T17:41:59.108547-06:00 START walking-anime-ten-seconds-77

2026-11-08T17:44:13.637045-06:00 Cooling/walking controller stopped: RuntimeError('Cooling experiment stopped: []')

Anime preview1 complete512x288,33frames20steps: visual inspection confirms2D anime style but character stands rather than walks. Rejected as walking deliverable; shown honestly as style preview. Stopped owned10s render rather than spend a long generation on that prompt. Verified90percent helper restored automatic fans+170W and exited0; no GPU render remains. New lightweight49-frame512x288 motion-first anime preview uses30steps,CFG6/shift8 (official Wan1.3B reference CFG6/shift8), explicit alternating steps/arm swing, negative standing/static, newseed30601661. Fresh local CPU encode, then one GPU renderer at original170W automaticfans with unchanged75/67/82C worker safeguards. No temporary administrator helper needed for this small preview. Cooling77C test already passed with all53PNG/latent/MP4 bytes identical: sampling280.0303s vs318.9209 (-12.20% time), cooling19.2587s vs47.5463 (-59.49% wait). Full10s not completed yet; do not represent preview as a finished long video.

Motion-preview startup corrected before GPU work: use absolute control/module paths rather than job-relative working-directory paths. Preserved the interrupted CPU-encode startup logs separately; GPU memory remains at desktop baseline.

Cooling measurements finalized:77/72C worker pilot at130W/90percentfan reduced sampling318.9209→280.0303s (-12.1944%), cooling47.5463→19.2587s (-59.4949%), measured stages -8.4666%. All candidate latents/53PNG/MP4bytes match reference exactly.42 CPU tests pass. Independent readback170W and both RTX fan policiesautomatic0. All temporary helpers exited successfully; currently only local motion-preview work runs under original power/fans and75/67/82C guard. No full10s anime completed; first style preview rejected for insufficient motion. New motion-first prompt/CFG6 shift8 usesofficialWanexample guidance; no newweights.

User rejected walking cycle in anime text-only preview (sliding/poor weight shift; head cropped). Do not scale up that clip. Switch to an experimental motion-reference video-to-video pass using existing better realistic walking frames, existing FP32 Wan VAE encode, then native ComfyUI sampler initialized from those encoded latents. No ControlNet/new weights/downloads; this is latent initialization, not guaranteed hard pose control. Short49-frame512x288 test,30steps,CFG6/shift8,denoise0.12: choose low schedule fraction because shift8 makes nominal0.5 a high-noise start. Inspect actual sigma and visual gait/style before deciding. Original170W/automaticfans,75/67/82C safeguards for lightweight tests; no root cooling helper active.

Motion-reference latent test completed finite:49frames512x28830stepsCFG6shift8denoise0.12. Gait preserved but visual inspection remains photographic, so not accepted as anime. Existing FP32VAE encode6.55s; sampling94.56s including24.51s cooling, peakguard76C. Trying denoise0.30 with same encoded reference andtext, no redundant encoding/newweights. Requested separate explicit approval for official ComfyOrg VACE1.3B FP16 file4.31GB (SHA256640ccc0577e6a5d4bb15cd91b11b699ef914fc55f126c5a1c544e152130784f2); disk474GBfree, existingComfyWanVaceToVideo node present, existingtextencoder/VAE reusable. No VACE weights downloaded pending approval.

Anime motion tests: text-only candidate rejected by user for bad walk cycle; both reference-initialized passes(denoise0.12 and0.30) complete finite but remain photographic/insufficiently stylized, strongerpass distortsbackground. No accepted10s anime walk yet. All experiments preserved; do not claim style or motion solved. Current roothelpers/render workers all exited; RTX170W and automaticfan policies restored. Additional officialVACE4.31GB download remains unapproved/notstarted. Preparing motion-control inputs and preserving logs while waiting; no GPU display/driver/suspend changes, no sound mixed.

User explicitly approved4.31GBVACE download and instructed autonomous work while away. Started streamed download to separate.partial with fullSHA256 verification before atomic install. Existing model/originals untouched. Prepared17-frame3-step nativePyTorch probe, then49-frame30-stepINT8/cache-off anime preview only if probe passes. Both useUUID-pinnedRTX,1GiBGTXweightdonor,RAMfallback,2GiBdesktopreserve,75/67Cworker pacing,82Couterstop,10GiBRTX/5GiBGTX/8GiBRAM guards. Added bounded900s sampler runtime andSTOP cancellation. No fan/power/driver/display changes for these small tests.

VACE download verified4309519800bytes/SHA256640ccc0577e6a5d4bb15cd91b11b699ef914fc55f126c5a1c544e152130784f2 in62.47s. ExistingFP32VAE prepared49-frameedge/inactive controls and1-frameanime style latent, finite,12.00s,peak2.81GiB,no cooling. New small conditioning helper matches coreWanVaceToVideo tensors/masks/trim exactly with andwithout reference;4CPUintegrationtests pass inclinvalid/nonfinite inputs, inCPU-onlyComfyprocess. Initialtestsys.pathshadowingcore nodes corrected beforeGPUprobe. Started17frame3step PyTorchattention probe with1GiBGTXdonor, hooks onbothbase andVACEblocks. Rootpower/fanhelpernotneeded.

VACE17frame3stepprobe finite/pass:6.57s sampling,no cooling,max59C,3.44GiBpeakallocated,RTX3244705920bytes/GTX1063784960bytesweights.49frame30step INT8withstyleimage finished149.07s incl25.02scooling,guardmax75C,monitoredRTX~4.9GBGTX1.18GB; visualQA rejects solarizedcolors/textures. Cross-checkofficialcoreV2Vtemplate usesbuiltinKorniaCanny0.4/0.8, notPillowFIND_EDGESapproximation. PreparedexactCPU-onlyCannyframes usinginstalledKornia (no dependenciesdownloaded); testingnew49framebaselinewithnativePyTorchattention,no styleimage, preserving all failedoutputs. No fan/power changes.

NativeCanny0.4/0.8 +PyTorch/noappearance-reference VACE49-frame30-step clip finished finite:189.73s sampling incl50.54scooling,maxguard76C. VisualQA nowshowsfull-bodyanimeandreferencewalk, butground/backgrounddistortion persists; notfinal. This comparison changedpreprocess,attention,andrefimagetogether, so do not blameonefactorforfirstsolarization. Preparedmatched0.25GiBdonor trial with exactsamecontrols,prompt,seed,steps,PyTorch and75/67Cguard; current~4.8GiBRTXusage leaves roomwhilekeeping2GiBdesktopreserve. Also preparedCPU-onlybody-focusedCanny guide withlow0.08/high0.16subjectregionx120:340 andofficial0.4/0.8background, torestorelowcontrastlegs/feet; no newpose model. Alltemporarypower/fansremainoriginal170W/automatic.

VACE matchedsmall-donor result:0.25GiB produces identicallatent/all49PNG/MP4bytes but sampling206.53s vs1GiB189.73s; cooling84.56s vs50.54s. Rootless77/72Cworker with stricter80Ccutoff/0.5soutermonitor completes189.89s,max78C,cooling67.53s,alsoall49framesbitidentical. This does not beat the cooler1GiB75/67Cbaseline, so do not adoptrootlesshigher-temperaturevariant.43CPUunit tests plus4nativeCPUconditioningparitytests pass. Continuingqualityrefinementwith1GiBdonor/75-67-82C,PyTorch,subject-focusededgeguides:removebackgroundroadlines that createfalseplatforms,retainlowcontrastlegsandfeet,fixedgroundline. BodyROI is manually tailored to this source, not universal pose estimation.

Subject-onlyCanny49-frame test completed252.89ssampling incl113.33scooling,max76C; visualQA rejectsnewfront-facing pose/poor side-onwalking despitecleanerbackground. Removing scene context allowed pose drift; preserve thisnegative result. Nextbounded17frame20-step VACEtest uses fullRGB walkingvideo conditioning (notlatentinitialization), reusingexisting encoded49-frameRGBsource+graycontroltensors. Testsstyle/motionpreservationwithoutthin-edgeinformationloss. No newdownloads,models,corechanges,power/fan changes.

RGBVACE17frame20stepprobe completesfinite40.67ssample,no cooling,max70C; remainsphotographic/saturated, notacceptedanime. Subject-onlyguidecausesfrontpose drift and rejected. Nextnative832x48017frame30stepfullsceneCannytest: modelofficialsupported480p, equivalenttokenbudgetto49frames512x288, same1GiBdonor/PyTorch/noapproximations/75-67-82Cguard. No longnativegenerationbeforevisualQA. Alloldoutputs preserved.

Native832x48017frame30step VACE test completedfinite223.40ssampling77.80scooling,maxguard76C,FP32controlencodepeak6.47GiB. VisualQA cleanerfull-bodyanime,reference-likealternatingsteps,andstableground inshortsample. Extendtobounded53frame3.3125s nativequalitytestusing53existingoriginalwalkingframes,notloops/interpolation. KeepPyTorch/1GiBdonor/30steps/cacheoff/75-67-82C. TightenphysicalmemoryguardstoRTX9800MiB/GTX4800MiB toretain>2GiB/>1GiBphysicalheadroom onreported11910/5964MiB devices;1smonitor,8GiBRAMstop,2400smaximumsampler. Do notstart10swithoutlongermotionreferenceandgaitQA.

Preparation sequencing fix: native480pCanny53 CPUjob was still generatingfiles whenVAEpreparerstarted; preparer failedbeforeinference onmissingframe028. Itsfailedattemptlogspreservedseparately; no generationstarted. WaitedforCPUjobexit0/all53framesbeforeproceeding andaddedsource-completenesspreflightbeforeCUDAinitialization. Also benchmarkingexistingnativeGPUCanny onthe same53frames withstrictpixelcomparison, sinceCPUedgepreparation is a headless-pipelinebottleneck (coreComfyCanny already usesGPU). No generatoroutputmath changed.

NativeGPUCannybenchmark53frames832x480:8.1609s,244.6MiBpeakallocated,guardmax58C,no cooling;0changedpixelsvsall53CPUguides. CoreCannyalreadyGPU-enabled; improvementappliesonlyourheadlesspreparation,notComfyUIcoreinferencespeed. Separate4-frameCPUserialrun13.0782safterimports (differentframecount,notdirectfullratio).53frameFP32VAEcontrolencodecompleted31.69sfinite,peak6.80GiB,headroomguardclean. Startedbounded53frame30stepnative832x480VACEgaitqualityrender; no frameupscaling/interpolation/looping, existing53walkingreferenceframes.

## Paused at user request — 2026-11-08T19:09:17.535000-06:00

User said continuetomorrow. Cooperatively stopped53frame native480psampler; controllerexited andownedCUDAworkerterminated. No furthergenerationwillstart. Onlydesktopportalcomputecontextremains. RTXreadback170W,bothfanpoliciesautomatic0;GTX130W. Savedconfigs,complete53framecontrols/embeddings,alltrialoutputs,failed/stoppedlogs anddetailedRESUME_TOMORROW.md. Theunfinished53framesamplingmustrestartfromseed; no resumablepartiallatentswerepersisted. Bestcompleteclip17frames832x480; no completed3sor10sanimeyet. Proceduralfoleyonlyseparatestems,notmixed.

## Resumed — 2026-11-09T10:50:56.247232-06:00

Userexplicitlyresumed. RTX29C/170W,GTX25C/130W;noGPUcomputejobs. Restart53frame832x48030stepVACEanimewalkfromsavedseed/verifiedcontrolsinafreshjob/outputdirectory;preservepreviousstoppedattemptandSTOP. Existingapprovedmodelsizeverified,nodownload/re-encoding. KeepRTXUUIDprimary,1GiBGTXweightdonor,PyTorch/cacheoff,allmemory/thermal/timeguards. No driver/display/suspend/power/fan changes.

### 2026-11-09 — Wan 2.2 Animate 14B feasibility and headless cleanup fix

User authorized the larger-model setup. Installed the official-tutorial-linked scaled FP8 Animate 14B and three native supporting encoders (26.66GB total), pinned and SHA256 verified; original models retained. Native CPU text/vision/VAE conditioning, nine-frame 144x256 one-step RTX sampling, GTX+RAM donor sampling and native CPU decoding passed. This is an inference smoke, not training or a final quality render.

Native RAM baseline: 9055MiB peak total RTX memory, 58C maximum monitored, 18.65s guarded stage. GTX+RAM: 6856MiB RTX and 2174MiB GTX, 62C, 21.85s. Identical finite latents, maximum difference zero. Donor saves primary memory but did not speed up this tiny test. Exact UUID mapping, 3GiB core reserve, independent memory/RAM/thermal guards; no driver, display, power or fan changes.

The donor probe exposed absent `PromptServer.instance` during headless cache cleanup. Three cleanup functions now safely handle an uninitialized server. Added four CPU regression cases; all 47 CPU safety tests passed. Installed extension backed up before the fix. Core unchanged. Full two-person character replacement, full-resolution rendering and actual training remain unvalidated. See WAN_ANIMATE_14B.md and benchmarks/results/rtx3060-wan22-animate14b.json.


## 2026-11-09 — preserve mapped offload weights

DisTorch2 now places unmodified modules directly on their assigned device, preserving existing compute-device cast flags and skipping no-op transfers. CPU weights supplied as mapped tensors stay mapped instead of staging through CUDA and back. Patched or wrapped weights retain the previous staging path. A matched nine-frame/one-step fixture remained bit-identical (max absolute difference and RMSE zero), including a zero bias patch and identity weight wrapper. All existing 47 CPU tests passed. This validates placement behavior, not animation quality or a general speed gain. The original source was backed up before installation.

## 2026-11-09 — isolate heavy Wan jobs

Added standalone user-systemd launcher and worker limit verification under tools/. A fixed service name rejects concurrent jobs; plan-only is the default. Render limits for this 64GB RAM machine are 36GiB high / 40GiB maximum, zero swap and an 1800-second timeout, with group OOM termination. Preparation and downloads have lower caps. Workers must still monitor available RAM/pressure, total GPU memory and temperatures.53 CPU checks passed, including six new isolation/launcher tests. Live 128MiB maximum / 96MiB high / no-swap / OOM-group-1 verification passed; a second job was refused while an existing worker PID remained unchanged. No real power/fan changes occur in the mocked unit tests.

## 2026-11-09 — validated short VACE cats preview

Added a VACE 1.3B profile, reproduction notes and measurement record for 17 frames at 288x512 and 16fps, 30 UniPC steps, CFG 6, shift 8. Native ComfyUI conditioning with the same local reference/text/pose data produced photographic cats and stable background without severe neon breakup. Guarded sampling took 124.90 seconds, with RTX 2374MiB, GTX 1191MiB and 76C maxima; GTX was a weight donor, RTX remained compute/display. No user media, weights or tensor fixtures are committed.

The local project now uses this working VACE preview. Existing Animate real-video tests remain rejected even with original weights, FP32 text/compute, alternate attention, synchronous offload, neutral face and protected background. A paired human control retained recognizable motion but still showed graphic/background artifacts. The exact underlying Animate defect is unresolved; no hardware/driver defect is established. The known-background test preserved outer regions but left the generated cats region corrupt with seams, so it is not offered as a quality fix. Longer motion, paw anatomy, identity and training remain unvalidated. Primary total-memory guard is now 9216MiB to retain 3GiB of the 12GiB display GPU. Core ComfyUI, drivers, display assignment, suspend, power and fans are unchanged.
