<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 10. Do not use as a complete reproduction guide. -->

# T800 Upright Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select the clean T800 retarget clips, align PPO speed commands with their motion range, and retrain an upright locomotion policy.

**Architecture:** Keep the original coupled GMR root pose untouched and build an immutable five-clip quality-filtered dataset. Retrain prior/PPO only after clip identity, tilt, foot appearance, and joint-limit gates pass.

**Tech Stack:** Python 3.13, NumPy, SciPy Rotation/Savitzky–Golay, pytest, official GMR/Mink/MuJoCo, SMP diffusion prior, RSL-RL PPO.

---

### Task 1: Root-orientation correction API

**Files:**
- Modify: `<SERVER_USER_DIR>/GMR_engineai_work/scripts/bvh_to_t800_csv.py`
- Test: `<SERVER_USER_DIR>/GMR_engineai_work/tests/test_bvh_to_t800_csv.py`

- [ ] Add a failing test proving that `apply_reference_root_orientation(qpos, reference_csv)` copies only the root orientation, converts CSV `xyzw` to MuJoCo `wxyz`, normalizes it, and leaves root position plus all 25 joints unchanged.
- [ ] Run `.venv/bin/pytest tests/test_bvh_to_t800_csv.py -q` and confirm failure because the function is absent.
- [ ] Implement the minimal function with shape, finite-value, unit-norm, and row-count validation.
- [ ] Run the focused test and the complete GMR suite; require all tests to pass.

### Task 2: Integrate reference CSV by filename

**Files:**
- Modify: `<SERVER_USER_DIR>/GMR_engineai_work/scripts/bvh_to_t800_csv.py`
- Test: `<SERVER_USER_DIR>/GMR_engineai_work/tests/test_bvh_to_t800_csv.py`

- [ ] Add a failing integration test for `--reference-root-dir`: matching `walk3_subject5.bvh` to `walk3_subject5.csv`, rejecting missing files and mismatched frame counts.
- [ ] Run the focused test and confirm the expected failure.
- [ ] Add the CLI option and apply the reference orientation before full-state smoothing.
- [ ] Run the focused and complete suites; require all tests to pass.

### Task 3: Reject the root-orientation probe

**Files:**
- Create outputs under: `<SERVER_USER_DIR>/GMR/output/t800_engineai_upright_probe`

- [ ] Regenerate `walk3_subject5` and `walk4_subject1` with the reference-root option.
- [ ] Measure quaternion norm, G1↔T800 pitch error, joint limit margin, and maximum joint step.
- [ ] Render side-by-side G1/T800 videos and inspect beginning, middle, and end frames.
- [x] Reject the probe because correcting rotation without recomputing translation leaves the T800 feet visibly airborne.

### Task 4: Clean Walk5 dataset

**Files:**
- Create dataset: `<SERVER_USER_DIR>/smp/datasets/csv/t800_engineai_walk5_clean_guard85`
- Create NPZ: `<SERVER_USER_DIR>/smp/datasets/npz/t800_engineai_walk5_clean_guard85`

- [ ] Copy only `walk1_subject1`, `walk1_subject2`, `walk1_subject5`, `walk3_subject1`, and `walk3_subject2` from the original authoritative GMR output.
- [ ] Verify exact file identity, row count, 32 columns, finite values, unit quaternions, tilt thresholds, 0 hard/soft/85%-guard violations, and bounded joint steps.
- [ ] Convert from 30 Hz CSV to 50 Hz, 10×55 NPZ windows.
- [ ] Validate NPZ shapes and compute new normalization statistics.

### Task 5: Speed-command alignment

**Files:**
- Modify: `<SERVER_USER_DIR>/smp/src/smp/rl/tasks/steering/forward_env_cfg.py`
- Test: the existing SMP task configuration tests or a new focused config test.

- [ ] Add a failing test asserting the T800 Walk12 task uses 0.5–2.0 m/s while the original G1 task remains unchanged.
- [ ] Introduce a T800-specific forward config instead of changing the shared G1 task globally.
- [ ] Run configuration and full SMP tests.

### Task 6: Retrain and final evaluation

**Files:**
- Create prior logs under: `<SERVER_USER_DIR>/smp/logs/pretrain/t800_loco_engineai_walk12_upright_guard85`
- Create PPO logs under: `<SERVER_USER_DIR>/smp/logs/rsl_rl/smp_forward_t800_engineai_upright_guard85`

- [ ] Train the SMP prior for 1,000 epochs and verify its final checkpoint plus finite validation loss.
- [ ] Run a 2-iteration PPO smoke test.
- [ ] Train PPO for 30,000 iterations with checkpoints every 500 iterations.
- [ ] Render the final checkpoint, extract key frames, and compare torso attitude against the corrected reference.
- [ ] Deliver the final strategy, policy video, training curve, and concise result summary.
