<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 43. Do not use as a complete reproduction guide. -->

# RoboStriker G1 Stage-3 Inference Reproduction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the official RoboStriker Preview G1-vs-G1 Stage-3 checkpoint in one headless Isaac Lab environment and produce a verified 30-second combat video plus an evidence log.

**Architecture:** Keep the reproduction isolated in `<SERVER_USER_DIR>/RoboStriker-Preview` and `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902`. Start by reusing the already working Isaac runtime through explicit environment variables, without installing into or changing the existing task-b package set. Apply only compatibility fixes proven necessary by failing checks; do not change policy dimensions, rewards, actions, robot dynamics, or checkpoint weights.

**Tech Stack:** Ubuntu 22.04, NVIDIA RTX 5090, Isaac Sim/Isaac Lab, Python, Gymnasium, PyTorch, RSL-RL, Git, FFmpeg.

---

### Task 1: Pin and inventory official inputs

**Files:**
- Inspect: `<SERVER_USER_DIR>/RoboStriker-Preview/README.md`
- Inspect: `<SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/unitree_description.tar.gz`
- Inspect: `<SERVER_USER_DIR>/RoboStriker-Preview/logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/combat/model_3000.pt`
- Create: `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/input_manifest.txt`

- [ ] **Step 1: Confirm the official Git revision**

Run:

```bash
git -C <SERVER_USER_DIR>/RoboStriker-Preview rev-parse HEAD
```

Expected: `52f54137770ff632391dd091e973ceb098ba3f98`.

- [ ] **Step 2: Materialize only the official checkpoint and Unitree asset archive**

Run:

```bash
cd <SERVER_USER_DIR>/RoboStriker-Preview
git checkout HEAD -- logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/combat/model_3000.pt
git checkout HEAD -- source/whole_body_tracking/unitree_description.tar.gz
```

Expected: both files exist and are non-empty; no motion-data download is required yet.

- [ ] **Step 3: Record immutable input evidence**

Run:

```bash
mkdir -p <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence
sha256sum \
  <SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/unitree_description.tar.gz \
  <SERVER_USER_DIR>/RoboStriker-Preview/logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/combat/model_3000.pt \
  > <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/input_manifest.txt
git -C <SERVER_USER_DIR>/RoboStriker-Preview status --short \
  >> <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/input_manifest.txt
```

Expected: two SHA256 lines followed by repository status.

### Task 2: Install official G1 assets without overwriting source

**Files:**
- Inspect: archive member list in `<SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/unitree_description.tar.gz`
- Create: `<SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/whole_body_tracking/assets/`

- [ ] **Step 1: Inspect archive paths and reject unsafe members**

Run:

```bash
tar -tzf <SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/unitree_description.tar.gz \
  > <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/unitree_archive_members.txt
python - <<'PY'
from pathlib import PurePosixPath
members = Path('<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/unitree_archive_members.txt').read_text().splitlines()
bad = [m for m in members if PurePosixPath(m).is_absolute() or '..' in PurePosixPath(m).parts]
assert not bad, bad
print(f'safe_members={len(members)}')
PY
```

Expected: `safe_members` is positive and no assertion occurs.

- [ ] **Step 2: Extract into a staging directory**

Run:

```bash
mkdir -p <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/assets_staging
tar -xzf <SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/unitree_description.tar.gz \
  -C <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/assets_staging
find <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/assets_staging -type f \
  | head -50
```

Expected: a G1 URDF and its referenced meshes are present.

- [ ] **Step 3: Place assets at the exact path expected by `robots/g1.py`**

Copy only after verifying the staging layout. The required final file is:

```text
<SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf
```

Expected: `test -s` succeeds for the URDF and every mesh path referenced by it resolves.

### Task 3: Prove and minimally fix Preview integration defects

**Files:**
- Modify only if the corresponding check fails: `<SERVER_USER_DIR>/RoboStriker-Preview/scripts/rsl_rl/play_selfplay.py`
- Modify only if inference actually requires absent motions: `<SERVER_USER_DIR>/RoboStriker-Preview/source/whole_body_tracking/whole_body_tracking/rsl_rl/rsl_rl/runners/on_policy_runner_stage3.py`
- Create: `<SERVER_USER_DIR>/RoboStriker-Preview/tests/test_stage3_preview_contract.py`

- [ ] **Step 1: Write a contract test for task registration**

The test must assert that `play_selfplay.py` imports `whole_body_tracking.tasks` before the `@hydra_task_config(...)` decorator. This catches the Preview bug without launching Isaac Sim.

- [ ] **Step 2: Run the test and verify the current source fails**

Run:

```bash
<SERVER_USER_DIR>/isaac_env/bin/python -m unittest -v tests/test_stage3_preview_contract.py
```

Expected before the fix: failure stating the task-registration import is absent.

- [ ] **Step 3: Add the minimal registration import**

Add after Isaac Lab is available and before the Hydra decorator:

```python
import whole_body_tracking.tasks  # noqa: F401
```

Do not edit any registered task IDs.

- [ ] **Step 4: Re-run syntax and contract checks**

Run:

```bash
find scripts source/whole_body_tracking/whole_body_tracking -name '*.py' -print0 \
  | xargs -0 <SERVER_USER_DIR>/isaac_env/bin/python -m py_compile
<SERVER_USER_DIR>/isaac_env/bin/python -m unittest -v tests/test_stage3_preview_contract.py
```

Expected: Python compilation succeeds and the contract test passes.

### Task 4: Run a one-environment startup smoke test

**Files:**
- Create: `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/logs/stage3_startup_smoke.log`
- Create: `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/logs/stage3_startup_smoke.status`

- [ ] **Step 1: Launch the official self-play task with the official checkpoint**

Use explicit `PYTHONPATH`, `LD_LIBRARY_PATH`, `ACCEPT_EULA=Y`, and `CUDA_VISIBLE_DEVICES=0`. Run:

```bash
cd <SERVER_USER_DIR>/RoboStriker-Preview
<SERVER_USER_DIR>/isaac_env/bin/python scripts/rsl_rl/play_selfplay.py \
  --task=Combat-Stage3-Selfplay-BoxingRing \
  --num_envs=1 \
  --headless \
  --load_run=combat \
  --checkpoint=model_3000.pt \
  --video \
  --video_length=100 \
  --episode_length_s=10 \
  --global_view
```

Expected: Isaac Sim launches, the G1 asset converts, the environment has `robot_A` and `robot_B`, and the exact `model_3000.pt` path appears in the log.

- [ ] **Step 2: Classify any failure before editing**

Check, in order: missing asset path, Isaac Lab API incompatibility, task registration, checkpoint tensor-shape mismatch, then optional AMP motion loading. Make only one change for the first observed failure and repeat the same smoke command.

- [ ] **Step 3: Record the successful runtime**

Run:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader
```

Expected: RTX 5090 is visible and there is no CUDA OOM, `Traceback`, `RuntimeError`, NaN, or checkpoint shape mismatch in the final smoke log.

### Task 5: Generate and verify the 30-second official G1-vs-G1 video

**Files:**
- Create: `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/videos/robostriker_g1_stage3_model3000_30s.mp4`
- Create: `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/video_probe.json`
- Create: `<LOCAL_WORKSPACE>\robostriker_g1_stage3_model3000_30s.mp4`

- [ ] **Step 1: Re-run inference for 30 seconds**

Use the same passing command from Task 4 with `--video_length=1500` and `--episode_length_s=40`.

Expected: one MP4 is produced by Gymnasium `RecordVideo`.

- [ ] **Step 2: Verify media integrity**

Run:

```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_read_frames \
  -show_entries format=duration,size \
  -of json /absolute/path/to/generated.mp4 \
  > <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/video_probe.json
```

Expected: H.264-compatible video, positive resolution, about 30 seconds, and all frames decode.

- [ ] **Step 3: Perform behavior gate**

Inspect start, 5 s, 10 s, 15 s, 20 s, 25 s, and end. Pass only if two G1 robots are visible, both remain simulated, at least one changes behavior relative to the opponent, and the video is not a frozen pose or immediate repeated reset. Record falls and contacts as observations; they do not invalidate the reproduction unless the policy never produces an interaction.

- [ ] **Step 4: Copy the final video to the local workspace**

Expected local path:

```text
<LOCAL_WORKSPACE>\robostriker_g1_stage3_model3000_30s.mp4
```

### Task 6: Preserve evidence and update project memory

**Files:**
- Modify: `<LOCAL_WORKSPACE>\T800_CONTEXT.md`
- Create: `<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/final_summary.txt`

- [ ] **Step 1: Record exact versions, command, fixes, and outcome**

Include official commit, checkpoint SHA256, asset SHA256, Isaac runtime version/commit, exact command, log path, video path, behavior findings, and any remaining gap from the official Isaac Lab 2.1.0 stack.

- [ ] **Step 2: Run final evidence checks**

Run:

```bash
grep -En 'Traceback|CUDA out of memory|RuntimeError|shape mismatch|NaN' \
  <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/logs/*.log || true
test -s <SERVER_USER_DIR>/robostriker_stage3_repro_20260902/evidence/video_probe.json
test -s <SERVER_USER_DIR>/RoboStriker-Preview/logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/combat/model_3000.pt
```

Expected: no fatal-error lines in the passing run and all required evidence files are non-empty.
