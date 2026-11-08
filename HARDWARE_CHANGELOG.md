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
