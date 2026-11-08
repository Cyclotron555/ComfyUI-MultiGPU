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
