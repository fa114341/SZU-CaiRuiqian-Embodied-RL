<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 55. Do not use as a complete reproduction guide. -->

# T800 Forward-Lateral Dodge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the working T800 forward policy and build a separate state-oracle dodge policy that faces forward, may decelerate, and sidesteps left or right before automatically returning to walking.

**Architecture:** Curate independently captured left and right grounded lateral segments from the existing LaFAN1 human BVHs and retarget both directions directly to T800.  Give the existing T800 state-oracle dodge task a dedicated forward/lateral AMP prior and command range, validate it with smoke/probe runs, and integrate it with the existing forward policy through a hysteretic two-policy switch.

**Tech Stack:** Python 3.11, NumPy, SciPy, LaFAN1 BVH, GMR/Mink/MuJoCo, MjLab, RSL-RL AMP/PPO, pytest, tmux, TensorBoard, ffmpeg/OpenCV.

**Repository note:** The local workspace and `GMR-engineai-work` have no Git metadata.  Record SHA-256 hashes and write isolated outputs instead of initializing or committing a repository.  The remote project may contain unrelated user changes; never commit them implicitly.

---

### Task 1: Curate grounded human side-step segments

**Files:**
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/scripts/select_lateral_bvh.py`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/left/walk1_subject2_f960_1050.bvh`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/left/walk1_subject5_f1380_1470.bvh`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/right/walk1_subject5_f3870_3960.bvh`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/right/walk2_subject3_f7020_7065.bvh`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/right/walk3_subject2_f4740_4785.bvh`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/metrics/source_selection.json`
- Test: `<SERVER_USER_DIR>/task_c_t800_dodge/tests/test_select_lateral_bvh.py`

- [ ] **Step 1: Write tests for the selection metrics**

Test a synthetic root path and shoulder heading.  The selector must compute
lateral displacement in the body frame, reject more than 10 degrees of heading
change, and accept speeds in `0.2..0.55 m/s`.

```python
def test_lateral_window_metrics():
    hips = np.column_stack((np.linspace(0.0, -0.9, 90), np.zeros(90)))
    left = np.tile([-0.1, 0.0], (90, 1))
    right = np.tile([0.1, 0.0], (90, 1))
    metrics = lateral_window_metrics(hips, left, right, fps=30.0)
    assert metrics["lateral_displacement_m"] == pytest.approx(-0.9)
    assert metrics["forward_displacement_m"] == pytest.approx(0.0)
    assert metrics["heading_change_deg"] == pytest.approx(0.0)
    assert metrics["lateral_speed_mps"] == pytest.approx(0.9 / (89 / 30))
    assert passes_lateral_gate(metrics)
```

- [ ] **Step 2: Run the test and verify RED**

```bash
cd <SERVER_USER_DIR>/task_c_t800_dodge
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_select_lateral_bvh.py
```

Expected: import failure because `select_lateral_bvh.py` does not exist.

- [ ] **Step 3: Implement deterministic metrics and BVH frame extraction**

Use the existing LaFAN reader, calculate the horizontal body-lateral axis from
`RightShoulder - LeftShoulder`, and preserve the source BVH hierarchy while
replacing the `Frames:` count and motion rows for `[start:end)`.

```python
def lateral_window_metrics(hips, left_shoulder, right_shoulder, fps):
    lateral = right_shoulder[0] - left_shoulder[0]
    lateral = lateral / np.linalg.norm(lateral)
    forward = np.array([-lateral[1], lateral[0]])
    displacement = hips[-1] - hips[0]
    end_lateral = right_shoulder[-1] - left_shoulder[-1]
    end_lateral = end_lateral / np.linalg.norm(end_lateral)
    heading = np.degrees(np.arccos(np.clip(lateral @ end_lateral, -1.0, 1.0)))
    duration = (len(hips) - 1) / fps
    return {
        "lateral_displacement_m": float(displacement @ lateral),
        "forward_displacement_m": float(displacement @ forward),
        "heading_change_deg": float(heading),
        "lateral_speed_mps": float(abs(displacement @ lateral) / duration),
    }


def passes_lateral_gate(metrics):
    return (
        0.2 <= metrics["lateral_speed_mps"] <= 0.55
        and abs(metrics["forward_displacement_m"]) <= 0.35
        and metrics["heading_change_deg"] <= 10.0
    )
```

- [ ] **Step 4: Run GREEN and extract the two discovered candidates**

```bash
cd <SERVER_USER_DIR>/task_c_t800_dodge
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_select_lateral_bvh.py
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python scripts/select_lateral_bvh.py \
  --input-dir <SERVER_USER_DIR>/GMR/motion_data/lafan1_walk12 \
  --output-dir source_bvh/left --fps 30
```

Expected: all five named segments pass the numeric gate and are recorded in
`metrics/source_selection.json`.

- [ ] **Step 5: Record source hashes**

```bash
find <SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/left -type f -name '*.bvh' -print0 \
  | sort -z | xargs -0 sha256sum > <SERVER_USER_DIR>/task_c_t800_dodge/metrics/source_sha256.txt
```

---

### Task 2: Verify independent left/right source coverage

**Files:**
- Read: `<SERVER_USER_DIR>/task_c_t800_dodge/metrics/source_selection.json`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/metrics/direction_coverage.json`

- [ ] **Step 1: Check the selected direction signs**

Assert both `left` records have negative body-lateral displacement and both
`right` records have positive body-lateral displacement.

- [ ] **Step 2: Check duration and speed coverage**

Require every clip to be at least 1.4 seconds and cover `0.2..0.55 m/s` lateral
speed without exceeding 10 degrees of heading change.

- [ ] **Step 3: Record the coverage result**

Write per-side clip count, speed minimum/maximum, duration minimum/maximum and
heading maximum to `metrics/direction_coverage.json`.

- [ ] **Step 4: Preserve the rejected mirror diagnostic**

Move the fixed-axis mirror experiment under `diagnostics/rejected_mirror/` and
exclude it from every GMR and NPZ glob.  Its body-relative direction sign did
not reverse, so it is evidence rather than training data.

---

### Task 3: Retarget human side steps directly to T800 and enforce data gates

**Files:**
- Read: `<SERVER_USER_DIR>/GMR_engineai_work/scripts/bvh_to_t800_csv.py`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/csv/left/*.csv`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/csv/right/*.csv`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/metrics/t800_csv_qa.json`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/videos/gmr/*.mp4`

- [ ] **Step 1: Retarget all four human clips with the established T800 configuration**

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=. <SERVER_USER_DIR>/GMR/.venv/bin/python scripts/bvh_to_t800_csv.py \
  --input-dir <SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/left \
  --output-dir <SERVER_USER_DIR>/task_c_t800_dodge/csv/left --pattern '*.bvh'
PYTHONPATH=. <SERVER_USER_DIR>/GMR/.venv/bin/python scripts/bvh_to_t800_csv.py \
  --input-dir <SERVER_USER_DIR>/task_c_t800_dodge/source_bvh/right \
  --output-dir <SERVER_USER_DIR>/task_c_t800_dodge/csv/right --pattern '*.bvh'
```

Expected: four T800 CSV files, produced without a G1 intermediate.

- [ ] **Step 2: Run existing joint-limit and finite-value checks**

Use the T800 model limits at 85% soft range.  Fail the task if any CSV contains
NaN/Inf, discontinuities above the existing guard threshold, or an out-of-order
25-joint header.

- [ ] **Step 3: Measure T800 root/contact/heading quality**

Record minimum foot height, per-foot ground-contact ratio, pelvis pitch median
and p95, heading change, lateral displacement and lateral speed in
`metrics/t800_csv_qa.json`.

- [ ] **Step 4: Render human/T800 overlays and T800-only videos**

Expected visual gate: grounded side stepping, forward-facing torso, no forearm
inversion, no foot floating and no joint-limit saturation.  Reject a clip before
NPZ conversion if either numeric or visual gate fails.

---

### Task 4: Build a dedicated T800 forward-lateral AMP prior

**Files:**
- Create: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/src/assets/motions/t800/amp/ForwardLateralDodge/*.npz`
- Modify: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/src/tasks/amp_loco/config/t800/env_cfgs.py`
- Modify: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/src/tasks/amp_loco/config/t800/dodge_env_cfgs.py`
- Modify: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/src/tasks/amp_loco/config/t800/dodge_rl_cfg.py`
- Modify: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/tests/tasks/test_t800_dodge_cfg.py`

- [ ] **Step 1: Add a failing configuration contract**

```python
[REDACTED: credential or access instruction]
    from src.tasks.amp_loco.config.t800.dodge_env_cfgs import (
        t800_amp_dodge_mimickit_flat_env_cfg,
    )
    from src.tasks.amp_loco.config.t800.dodge_rl_cfg import (
        t800_amp_dodge_mimickit_ppo_runner_cfg,
    )

    env = t800_amp_dodge_mimickit_flat_env_cfg()
    agent = t800_amp_dodge_mimickit_ppo_runner_cfg()
    twist = env.commands["twist"]
    assert twist.heading_command is False
    assert twist.ranges.lin_vel_x == (0.2, 0.6)
    assert twist.ranges.lin_vel_y == (-0.5, 0.5)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert agent.amp_motion_files.endswith("motions/t800/amp/ForwardLateralDodge")
```

- [ ] **Step 2: Run the contract and verify RED**

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-t800-work
.venv/bin/python -m pytest -q tests/tasks/test_t800_dodge_cfg.py
```

- [ ] **Step 3: Convert the four accepted lateral CSV files**

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-t800-work
mkdir -p src/assets/motions/t800/amp/ForwardLateralDodge
for csv in <SERVER_USER_DIR>/task_c_t800_dodge/csv/left/*.csv <SERVER_USER_DIR>/task_c_t800_dodge/csv/right/*.csv; do
  .venv/bin/python scripts/csv_to_npz.py "$csv" \
    --output-dir src/assets/motions/t800/amp/ForwardLateralDodge
done
cp src/assets/motions/t800/amp/WalkAndStandPitch25GroundedV3/*.npz \
  src/assets/motions/t800/amp/ForwardLateralDodge/
```

- [ ] **Step 4: Add the dedicated prior constant and dodge overrides**

```python
T800_DODGE_MOTION_DIR = os.path.abspath(
    os.path.join(T800_MOTION_BASE, "ForwardLateralDodge")
)
```

In the dodge builder, point both reset events and the runner's
`amp_motion_files` to `T800_DODGE_MOTION_DIR`, disable heading commands, and set
the exact forward/lateral/yaw ranges asserted by the test.

- [ ] **Step 5: Run GREEN and replay every NPZ**

```bash
.venv/bin/python -m pytest -q tests/tasks/test_t800_dodge_cfg.py tests/scripts/test_csv_to_npz_t800.py
```

Expected: all tests pass and every NPZ replay matches its CSV direction.

---

### Task 5: Smoke and probe the state-oracle dodge policy

**Files:**
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/logs/smoke16_iter2.log`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/logs/probe256_iter1000.log`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/videos/probe/*.mp4`
- Read: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/scripts/train.py`
- Read: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/scripts/play.py`

- [ ] **Step 1: Confirm training tmux and GPU headroom**

```bash
tmux has-session -t t800-loco-groundedv3-home10145-resume8192-to60000
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Do not stop the locomotion run.  If free memory is below the measured smoke
requirement plus 2 GiB, wait for natural completion.

- [ ] **Step 2: Run a 16-environment, 2-iteration smoke**

Use task `EngineAI-T800-AMP-Dodge-MimicKit-Flat`, official PPO defaults, an
isolated run name and no checkpoint resume.  Require two completed iterations
and no Traceback, CUDA OOM, RuntimeError or NaN.

- [ ] **Step 3: Run a 256-environment, 1000-iteration probe**

Save every 250 iterations.  Compare the first and last 100 iterations for mean
reward, episode length, collision/termination rate, minimum clearance and
lateral displacement.

- [ ] **Step 4: Record left/right/no-obstacle videos**

Require upright survival, visible lateral motion in both directions, no frozen
pose and no regression of forward walking.  Do not start a formal run unless the
probe passes these qualitative and quantitative gates.

---

### Task 6: Train and evaluate the formal dodge policy

**Files:**
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/logs/formal4096_iter20000.log`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/metrics/formal_eval.json`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/videos/formal/*.mp4`

- [ ] **Step 1: Measure 4096-environment memory before launch**

Run a bounded 16/256 environment measurement and extrapolate conservatively.
Wait rather than stopping the locomotion tmux if resources are insufficient.

- [ ] **Step 2: Launch 4096 environments for 20,000 iterations**

Use official PPO defaults, the dedicated state-oracle dodge task, an isolated
name, TensorBoard logging and checkpoints at 2,000/5,000/10,000/20,000.

- [ ] **Step 3: Apply formal gates at each checkpoint**

Evaluate 20-second no-obstacle survival, left/right obstacle avoidance,
collision rate, minimum clearance, heading error, lateral displacement,
recovery time and termination reason distribution.

---

### Task 7: Integrate hysteretic walk/dodge switching

**Files:**
- Create: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/deploy/t800_policy_switch.py`
- Create: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/tests/deploy/test_t800_policy_switch.py`
- Modify: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/T800_CONTEXT.md`

- [ ] **Step 1: Write state-machine tests**

Cover WALK -> DODGE after three unsafe frames, DODGE -> WALK only after ten clear
frames, and no oscillation on alternating safe/unsafe inputs.

```python
def test_switch_uses_hysteresis():
    switch = PolicySwitch(unsafe_frames=3, clear_frames=10)
    assert [switch.update(True) for _ in range(2)] == [Mode.WALK, Mode.WALK]
    assert switch.update(True) is Mode.DODGE
    assert [switch.update(False) for _ in range(9)] == [Mode.DODGE] * 9
    assert switch.update(False) is Mode.WALK
```

- [ ] **Step 2: Implement the minimal state machine and run GREEN**

The switch consumes only `unsafe: bool` and exposes `Mode.WALK`/`Mode.DODGE`;
obstacle estimation stays outside this module.

- [ ] **Step 3: Run repeated walk -> dodge -> walk integration trials**

Require no reset, fall, reverse command or high-frequency switching.  Save the
checkpoint paths, configs, metrics, videos and hashes in `T800_CONTEXT.md`.

---

### Task 8: Final verification and handoff

**Files:**
- Modify: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/T800_CONTEXT.md`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/SOURCE_MANIFEST.json`
- Create: `<SERVER_USER_DIR>/task_c_t800_dodge/SHA256SUMS`

- [ ] **Step 1: Run focused tests**

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-t800-work
.venv/bin/python -m pytest -q \
  tests/tasks/test_t800_dodge_cfg.py \
  tests/scripts/test_csv_to_npz_t800.py \
  tests/deploy/test_t800_policy_switch.py
```

- [ ] **Step 2: Scan all final logs**

```bash
grep -R -nE 'Traceback|CUDA out of memory|RuntimeError|(^|[^A-Za-z])NaN([^A-Za-z]|$)' \
  <SERVER_USER_DIR>/task_c_t800_dodge/logs
```

Expected: no matches.

- [ ] **Step 3: Write manifests and hashes**

Include source identity, frame ranges, retargeting config, command ranges,
checkpoint lineage, environment/agent configs, metrics and video paths.  Exclude
all passwords, tokens and server addresses.
