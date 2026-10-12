<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 84. Do not use as a complete reproduction guide. -->

# T800 Celebration Arm Retarget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, action-scale-aware arm posture regularizer to direct LAFAN1→T800 GMR, select the weakest candidate that passes the data gates, then validate it with a fresh default-PPO probe without interrupting task a.

**Architecture:** Keep the official GMR frame targets, two-stage IK, 85% configuration limit, EngineAI environment, and PPO unchanged. Add one low-cost Mink `PostureTask` to both IK stages; its per-DoF cost is zero outside the arms and proportional to the inverse EngineAI action scale within the arms. Preserve old behavior when the new CLI strength is zero and use isolated output paths for all candidate data, videos, NPZ files, and checkpoints.

**Tech Stack:** Python 3.11, NumPy, MuJoCo, Mink IK, pytest, Isaac Sim/Isaac Lab, EngineAI RL Lab, RSL-RL PPO, TensorBoard, tmux, ffmpeg.

**Repository note:** `<LOCAL_WORKSPACE>` and `GMR-engineai-work` do not contain Git metadata. Commit steps are therefore replaced by SHA-256 snapshots and explicit file copies. Do not initialize a repository as part of this task.

---

### Task 1: Add failing unit tests for the arm posture cost

**Files:**
- Modify: `<LOCAL_WORKSPACE>\GMR-engineai-work\tests\test_bvh_to_t800_csv.py`
- Test: `<LOCAL_WORKSPACE>\GMR-engineai-work\tests\test_bvh_to_t800_csv.py`

- [ ] **Step 1: Extend imports for the desired API**

Add `SimpleNamespace`, `ROBOT_XML_DICT`, and the new functions to the test imports:

```python
from types import SimpleNamespace

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

from scripts.bvh_to_t800_csv import (
    add_arm_action_posture_task,
    arm_action_posture_cost,
    arm_action_posture_target,
    apply_human_elbow_bend,
    clip_limited_joints,
    configuration_limit_with_factor,
    qpos_to_csv,
    rotate_human_pose_z,
    scaled_joint_ranges,
    smooth_joint_positions,
    warm_start,
)
```

- [ ] **Step 2: Add tests that describe the cost-vector behavior**

Append these tests:

```python
def test_arm_action_posture_cost_is_zero_when_disabled():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))

    cost = arm_action_posture_cost(model, strength=0.0)

    np.testing.assert_array_equal(cost, np.zeros(model.nv))


[REDACTED: credential or access instruction]
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))

    cost = arm_action_posture_cost(model, strength=0.01)

    def dof_cost(joint_name):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return cost[model.jnt_dofadr[joint_id]]

    assert dof_cost("J00_HIP_PITCH_L") == pytest.approx(0.0)
    assert dof_cost("J13_SHOULDER_PITCH_L") == pytest.approx(0.01 / 0.2)
    assert dof_cost("J15_SHOULDER_YAW_L") == pytest.approx(0.01 / 0.05)
    assert dof_cost("J15_SHOULDER_YAW_L") == pytest.approx(
        4.0 * dof_cost("J13_SHOULDER_PITCH_L")
    )
    assert dof_cost("J20_SHOULDER_YAW_R") == pytest.approx(
        dof_cost("J15_SHOULDER_YAW_L")
    )


@pytest.mark.parametrize("strength", [-1.0, np.nan, np.inf])
def test_arm_action_posture_cost_rejects_invalid_strength(strength):
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))

    with pytest.raises(ValueError, match="arm posture strength"):
        arm_action_posture_cost(model, strength=strength)
```

- [ ] **Step 3: Add tests for installing the Mink posture task**

```python
def test_add_arm_action_posture_task_is_noop_when_disabled():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))
    retargeter = SimpleNamespace(model=model, tasks1=[], tasks2=[])

    task = add_arm_action_posture_task(retargeter, strength=0.0)

    assert task is None
    assert retargeter.tasks1 == []
    assert retargeter.tasks2 == []


[REDACTED: credential or access instruction]
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))
    retargeter = SimpleNamespace(model=model, tasks1=[], tasks2=[])

    task = add_arm_action_posture_task(retargeter, strength=0.01)

    assert retargeter.tasks1 == [task]
    assert retargeter.tasks2 == [task]
    np.testing.assert_allclose(task.target_q, arm_action_posture_target(model))


def test_arm_action_posture_target_matches_engineai_policy_defaults():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))

    target = arm_action_posture_target(model)

    def joint_target(joint_name):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return target[model.jnt_qposadr[joint_id]]

    assert joint_target("J14_SHOULDER_ROLL_L") == pytest.approx(0.15)
    assert joint_target("J16_ELBOW_PITCH_L") == pytest.approx(-0.25)
    assert joint_target("J19_SHOULDER_ROLL_R") == pytest.approx(-0.15)
    assert joint_target("J21_ELBOW_PITCH_R") == pytest.approx(-0.25)
    assert joint_target("J15_SHOULDER_YAW_L") == pytest.approx(0.0)
```

- [ ] **Step 4: Add a default-compatibility signature test**

```python
def test_arm_action_posture_is_disabled_by_default():
    default = inspect.signature(converter.retarget_bvh).parameters[
        "arm_posture_strength"
    ].default
    assert default == 0.0
```

- [ ] **Step 5: Upload only the test file and verify RED**

Run locally to compute the expected source hash, then upload to the server copy:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "<LOCAL_WORKSPACE>\GMR-engineai-work\tests\test_bvh_to_t800_csv.py"
[REDACTED: credential or access instruction]
```

Run on the server:

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_bvh_to_t800_csv.py
```

Expected: collection fails because `arm_action_posture_cost` and `add_arm_action_posture_task` do not exist. This is the required RED state.

---

### Task 2: Implement the minimal posture-cost API

**Files:**
- Modify: `<LOCAL_WORKSPACE>\GMR-engineai-work\scripts\bvh_to_t800_csv.py:5-21`
- Modify: `<LOCAL_WORKSPACE>\GMR-engineai-work\scripts\bvh_to_t800_csv.py:81-126`
- Test: `<LOCAL_WORKSPACE>\GMR-engineai-work\tests\test_bvh_to_t800_csv.py`

- [ ] **Step 1: Add MuJoCo and arm action-scale constants**

Add the import and constants:

```python
import mujoco as mj

T800_ARM_ACTION_SCALES = {
    "J13_SHOULDER_PITCH_L": 0.2,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.2,
    "J17_ELBOW_YAW_L": 0.05,
    "J18_SHOULDER_PITCH_R": 0.2,
    "J19_SHOULDER_ROLL_R": 0.2,
    "J20_SHOULDER_YAW_R": 0.05,
    "J21_ELBOW_PITCH_R": 0.2,
    "J22_ELBOW_YAW_R": 0.05,
}
T800_ARM_DEFAULT_JOINT_POSITIONS = {
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0,
    "J19_SHOULDER_ROLL_R": -0.15,
    "J20_SHOULDER_YAW_R": 0.0,
    "J21_ELBOW_PITCH_R": -0.25,
    "J22_ELBOW_YAW_R": 0.0,
}
```

- [ ] **Step 2: Implement the cost vector**

Insert after `configuration_limit_with_factor`:

```python
def arm_action_posture_cost(model: object, strength: float) -> np.ndarray:
    """Build a per-DoF arm posture cost normalized by policy action scale."""
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("arm posture strength must be finite and non-negative")
    cost = np.zeros(model.nv, dtype=float)
    for joint_name, action_scale in T800_ARM_ACTION_SCALES.items():
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing T800 arm joint: {joint_name}")
        dof_address = int(model.jnt_dofadr[joint_id])
        cost[dof_address] = strength / action_scale
    return cost


def arm_action_posture_target(model: object) -> np.ndarray:
    """Return a qpos target aligned with EngineAI policy default arm angles."""
    target = np.asarray(model.qpos0, dtype=float).copy()
    for joint_name, joint_position in T800_ARM_DEFAULT_JOINT_POSITIONS.items():
        joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"missing T800 arm joint: {joint_name}")
        target[int(model.jnt_qposadr[joint_id])] = joint_position
    return target
```

- [ ] **Step 3: Implement opt-in task installation**

```python
def add_arm_action_posture_task(retargeter: object, strength: float) -> object | None:
    """Add one action-aware arm posture task to both GMR IK stages."""
    cost = arm_action_posture_cost(retargeter.model, strength)
    if not np.any(cost):
        return None
    task = mink.PostureTask(
        retargeter.model,
        cost=cost,
        gain=1.0,
        lm_damping=1.0,
    )
    task.set_target(arm_action_posture_target(retargeter.model))
    retargeter.tasks1.append(task)
    retargeter.tasks2.append(task)
    return task
```

- [ ] **Step 4: Upload production code and verify GREEN for the focused tests**

```powershell
[REDACTED: credential or access instruction]
```

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_bvh_to_t800_csv.py -k 'arm_action_posture_cost or add_arm_action_posture_task'
```

Expected: all focused tests pass.

- [ ] **Step 5: Run the complete converter test file**

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_bvh_to_t800_csv.py
```

Expected: zero failures and zero errors.

---

### Task 3: Wire the regularizer into the converter and CLI

**Files:**
- Modify: `<LOCAL_WORKSPACE>\GMR-engineai-work\scripts\bvh_to_t800_csv.py:252-313`
- Modify: `<LOCAL_WORKSPACE>\GMR-engineai-work\scripts\bvh_to_t800_csv.py:316-352`
- Test: `<LOCAL_WORKSPACE>\GMR-engineai-work\tests\test_bvh_to_t800_csv.py`

- [ ] **Step 1: Verify the signature test is still RED**

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_bvh_to_t800_csv.py::test_arm_action_posture_is_disabled_by_default
```

Expected: failure because `retarget_bvh` has no `arm_posture_strength` parameter.

- [ ] **Step 2: Add the opt-in function parameter and install before warm-start**

Change the signature and setup:

```python
def retarget_bvh(
    input_path: Path,
    output_path: Path,
    max_frames: int | None = None,
    warmup_iterations: int = 50,
    source_yaw_degrees: float = 90.0,
    smoothing_window: int = 31,
    joint_limit_factor: float = 0.85,
    source_elbow_override: bool = False,
    reference_root_dir: Path | None = None,
    arm_posture_strength: float = 0.0,
) -> Path:
```

Immediately after installing the 85% configuration limit:

```python
    posture_task = add_arm_action_posture_task(
        retargeter, strength=arm_posture_strength
    )
    if posture_task is not None:
        active_costs = posture_task.cost[posture_task.cost > 0.0]
        print(
            "arm posture regularizer "
            f"strength={arm_posture_strength:g} "
            f"cost_min={active_costs.min():.6g} "
            f"cost_max={active_costs.max():.6g}",
            flush=True,
        )
```

- [ ] **Step 3: Add and forward the CLI parameter**

```python
    parser.add_argument("--arm-posture-strength", type=float, default=0.0)
```

Forward it in the call:

```python
            arm_posture_strength=args.arm_posture_strength,
```

- [ ] **Step 4: Upload and run signature plus full tests**

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python -m pytest -q tests/test_bvh_to_t800_csv.py
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work MUJOCO_GL=egl <SERVER_USER_DIR>/GMR/.venv/bin/python scripts/bvh_to_t800_csv.py --help
```

Expected: all tests pass and help contains `--arm-posture-strength`.

- [ ] **Step 5: Capture source snapshots**

```bash
sha256sum scripts/bvh_to_t800_csv.py tests/test_bvh_to_t800_csv.py
```

Save the hashes in `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/source_sha256.txt` using `sha256sum ... >` only after creating the isolated directory.

---

### Task 4: Prove backward compatibility and generate candidate CSV files

**Files:**
- Read: `<SERVER_USER_DIR>/task_b_t800_actions/source_bvh/celebration_dance1_subject2_f2880_3180.bvh`
- Read: `<SERVER_USER_DIR>/task_b_t800_actions/gmr_csv/celebration_dance1_subject2_f2880_3180.csv`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/*/celebration_dance1_subject2_f2880_3180.csv`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/gmr_*.log`

- [ ] **Step 1: Confirm task a and GPU state before every simulation batch**

```bash
tmux has-session -t t800-oldgmr-walk10-drop2-tilt-w05-free30-pipeline
pgrep -af 'Smp-Forward-T800-Walk12'
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
```

Expected: task a tmux and process remain present. Do not kill or suspend them.

- [ ] **Step 2: Create isolated output directories**

```bash
mkdir -p <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/{csv,logs,videos,npz,tmp}
```

- [ ] **Step 3: Generate a zero-strength compatibility CSV**

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work MUJOCO_GL=egl <SERVER_USER_DIR>/GMR/.venv/bin/python scripts/bvh_to_t800_csv.py \
  --input-dir <SERVER_USER_DIR>/task_b_t800_actions/source_bvh \
  --output-dir <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0 \
  --pattern 'celebration_dance1_subject2_f2880_3180.bvh' \
  --warmup-iterations 50 --source-yaw-degrees 90 --smoothing-window 31 \
  --joint-limit-factor 0.85 --arm-posture-strength 0 \
  2>&1 | tee <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/gmr_cost_0.log
```

Compare:

```bash
cmp <SERVER_USER_DIR>/task_b_t800_actions/gmr_csv/celebration_dance1_subject2_f2880_3180.csv <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0/celebration_dance1_subject2_f2880_3180.csv
```

Expected: exit code 0. If it differs, stop before candidate generation and investigate the baseline path/config mismatch.

- [ ] **Step 4: Generate one-variable candidates**

Run the same command separately with strengths `0.001`, `0.003`, `0.01`, `0.03`, and `0.1`, using matching directories and logs:

```bash
--output-dir <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0p001 --arm-posture-strength 0.001
--output-dir <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0p003 --arm-posture-strength 0.003
--output-dir <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0p01  --arm-posture-strength 0.01
--output-dir <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0p03  --arm-posture-strength 0.03
--output-dir <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_0p1   --arm-posture-strength 0.1
```

Expected for every candidate: 300 finite rows, no traceback, and no source/output overwrite.

---

### Task 5: Select the weakest candidate that passes the data gates

**Files:**
- Read: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/*/*.csv`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/candidate_metrics.txt`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/selected/celebration_dance1_subject2_f2880_3180.csv`

- [ ] **Step 1: Compute metrics using the real T800 MuJoCo model**

Use `<SERVER_USER_DIR>/GMR/.venv/bin/python` to load every CSV, convert root quaternion xyzw→wxyz, run `mujoco.mj_forward`, and record the metrics with this complete QA command:

```bash
cd <SERVER_USER_DIR>/GMR_engineai_work
PYTHONPATH=<SERVER_USER_DIR>/GMR_engineai_work <SERVER_USER_DIR>/GMR/.venv/bin/python - <<'PY' | tee <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/candidate_metrics.txt
from pathlib import Path
import mujoco
import numpy as np
from general_motion_retargeting.params import ROBOT_XML_DICT

root = Path("<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv")
selected_file = Path("<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/selected_candidate.txt")
selected_file.unlink(missing_ok=True)
candidates = [
    (0.0, root / "cost_0"),
    (0.001, root / "cost_0p001"),
    (0.003, root / "cost_0p003"),
    (0.01, root / "cost_0p01"),
    (0.03, root / "cost_0p03"),
    (0.1, root / "cost_0p1"),
]
stem = "celebration_dance1_subject2_f2880_3180.csv"
model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["engineai_t800"]))
data = mujoco.MjData(model)
arm_scales = {
    "J13_SHOULDER_PITCH_L": 0.2,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.2,
    "J17_ELBOW_YAW_L": 0.05,
    "J18_SHOULDER_PITCH_R": 0.2,
    "J19_SHOULDER_ROLL_R": 0.2,
    "J20_SHOULDER_YAW_R": 0.05,
    "J21_ELBOW_PITCH_R": 0.2,
    "J22_ELBOW_YAW_R": 0.05,
}

def qpos_address(joint_name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return int(model.jnt_qposadr[joint_id])

def load_qpos(path):
    csv = np.loadtxt(path, delimiter=",", ndmin=2)
    if csv.shape != (300, 32):
        raise ValueError(f"unexpected CSV shape {csv.shape} for {path}")
    qpos = csv.copy()
    qpos[:, 3:7] = csv[:, [6, 3, 4, 5]]
    return qpos

def wrist_positions(qpos):
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_L"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_WRIST_END_R"),
    ]
    positions = np.empty((len(qpos), 2, 3))
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        positions[frame] = data.xpos[ids]
    return positions

base_qpos = load_qpos(candidates[0][1] / stem)
base_wrists = wrist_positions(base_qpos)
leg_names = [f"J{i:02d}_{suffix}" for i, suffix in [
    (0, "HIP_PITCH_L"), (1, "HIP_ROLL_L"), (2, "HIP_YAW_L"),
    (3, "KNEE_PITCH_L"), (4, "ANKLE_PITCH_L"), (5, "ANKLE_ROLL_L"),
    (6, "HIP_PITCH_R"), (7, "HIP_ROLL_R"), (8, "HIP_YAW_R"),
    (9, "KNEE_PITCH_R"), (10, "ANKLE_PITCH_R"), (11, "ANKLE_ROLL_R"),
]]
leg_addresses = [qpos_address(name) for name in leg_names]
arm_addresses = {name: qpos_address(name) for name in arm_scales}
arm_defaults = {
    "J13_SHOULDER_PITCH_L": 0.0,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0,
    "J16_ELBOW_PITCH_L": -0.25,
    "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0,
    "J19_SHOULDER_ROLL_R": -0.15,
    "J20_SHOULDER_YAW_R": 0.0,
    "J21_ELBOW_PITCH_R": -0.25,
    "J22_ELBOW_YAW_R": 0.0,
}
yaw_names = ["J15_SHOULDER_YAW_L", "J20_SHOULDER_YAW_R"]

scaled_ranges = np.asarray(model.jnt_range, dtype=float).copy()
for joint_id in range(model.njnt):
    if model.jnt_limited[joint_id]:
        lo, hi = scaled_ranges[joint_id]
        mid = 0.5 * (lo + hi)
        half = 0.425 * (hi - lo)
        scaled_ranges[joint_id] = [mid - half, mid + half]

passing = []
print("strength finite limit_ok guard_L% guard_R% yaw_L_p95 yaw_R_p95 arm_penalty_p95 root_rms leg_rms wrist_L_p95 wrist_R_p95 pass")
for strength, directory in candidates:
    path = directory / stem
    qpos = load_qpos(path)
    finite = bool(np.isfinite(qpos).all())
    limit_ok = True
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        address = int(model.jnt_qposadr[joint_id])
        lo, hi = scaled_ranges[joint_id]
        limit_ok &= bool(np.all(qpos[:, address] >= lo - 1e-6))
        limit_ok &= bool(np.all(qpos[:, address] <= hi + 1e-6))

    normalized = {
        name: (qpos[:, address] - arm_defaults[name]) / arm_scales[name]
        for name, address in arm_addresses.items()
    }
    yaw_p95 = {name: float(np.percentile(np.abs(normalized[name]), 95)) for name in yaw_names}
    guard_pct = {}
    for name in yaw_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = scaled_ranges[joint_id]
        values = qpos[:, arm_addresses[name]]
        guard_pct[name] = 100.0 * float(np.mean(np.isclose(values, lo, atol=1e-6) | np.isclose(values, hi, atol=1e-6)))

    arm_matrix = np.column_stack([normalized[name] for name in arm_scales])
    arm_penalty = float(np.percentile(0.075 * np.sum(arm_matrix**2, axis=1), 95))
    root_rms = float(np.sqrt(np.mean((qpos[:, :3] - base_qpos[:, :3]) ** 2)))
    leg_rms = float(np.sqrt(np.mean((qpos[:, leg_addresses] - base_qpos[:, leg_addresses]) ** 2)))
    wrist_error = np.linalg.norm(wrist_positions(qpos) - base_wrists, axis=-1)
    wrist_p95 = np.percentile(wrist_error, 95, axis=0)
    passed = (
        finite and limit_ok
        and all(value < 5.0 for value in guard_pct.values())
        and all(value <= 25.0 for value in yaw_p95.values())
        and arm_penalty <= 75.0
        and root_rms <= 0.01
        and leg_rms <= 0.02
        and bool(np.all(wrist_p95 <= 0.10))
    )
    print(
        f"{strength:g} {finite} {limit_ok} "
        f"{guard_pct[yaw_names[0]]:.3f} {guard_pct[yaw_names[1]]:.3f} "
        f"{yaw_p95[yaw_names[0]]:.3f} {yaw_p95[yaw_names[1]]:.3f} "
        f"{arm_penalty:.3f} {root_rms:.6f} {leg_rms:.6f} "
        f"{wrist_p95[0]:.6f} {wrist_p95[1]:.6f} {passed}"
    )
    if passed and strength > 0.0:
        passing.append((strength, directory))

if passing:
    selected = min(passing, key=lambda item: item[0])[1]
    selected_file.write_text(str(selected) + "\n", encoding="utf-8")
    print(f"SELECTED {selected}")
else:
    print("SELECTED NONE")
PY
```

The command records:

```text
finite values
85%-guard equality count and frame percentage for J15/J20
abs((q-default)/scale) p95 for J15/J20
arm-only 0.075*sum(((q-default)/scale)^2) p95
root-position RMS versus cost_0
leg-joint RMS versus cost_0
left/right wrist 3-D FK deviation p95 versus cost_0
```

Write the complete table to `candidate_metrics.txt`. The joint columns are the numeric T800 SDK order in CSV: J00 begins at column 7; J15 and J20 are columns 22 and 27. The same read-only QA command must write the absolute directory of the weakest passing candidate as the only line of `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/selected_candidate.txt`; it must leave that file absent when no candidate passes.

- [ ] **Step 2: Apply all gates without manual exceptions**

The selected candidate must satisfy:

```text
J15/J20 guard-frame percentage < 5%
J15/J20 normalized-action p95 <= 25
arm reset action-rate penalty p95 <= 75
root position RMS <= 0.01 m
leg joint RMS <= 0.02 rad
left/right wrist FK deviation p95 <= 0.10 m
all values finite and inside the 85% limits
```

- [ ] **Step 3: Select the smallest passing strength**

Read the QA-selected directory, validate that it stays inside the isolated candidate root, then copy rather than move it:

```bash
read -r armreg_selected_dir < <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/selected_candidate.txt
case "$armreg_selected_dir" in
  <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/csv/cost_*) ;;
  *) echo "invalid selected candidate: $armreg_selected_dir" >&2; exit 1 ;;
esac
test -f "$armreg_selected_dir/celebration_dance1_subject2_f2880_3180.csv"
mkdir -p <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/selected
cp "$armreg_selected_dir/celebration_dance1_subject2_f2880_3180.csv" <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/selected/celebration_dance1_subject2_f2880_3180.csv
```

If `selected_candidate.txt` is absent, do not copy, render, convert, or train; report that posture regularization failed its data gate and return to the separately scoped shoulder-link mapping design.

---

### Task 6: Render and verify the selected retargeted motion

**Files:**
- Read: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/selected/celebration_dance1_subject2_f2880_3180.csv`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/videos/celebration_t800_armreg_selected_csv.mp4`
- Create: `<LOCAL_WORKSPACE>\task_b_motion_selection\videos\armreg_v1\celebration_t800_armreg_selected_csv.mp4`

- [ ] **Step 1: Render the 10-second T800 CSV**

```bash
cd <SERVER_USER_DIR>/smp
MUJOCO_GL=egl <SERVER_USER_DIR>/smp/.venv/bin/python scripts/render_t800_csv.py \
  --input-file <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/selected/celebration_dance1_subject2_f2880_3180.csv \
  --output-file <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/videos/celebration_t800_armreg_selected_csv.mp4 \
  --robot t800 --input-fps 30 --output-fps 30 --start-seconds 0 --duration-seconds 10 \
  --device cpu --width 640 --height 480 \
  2>&1 | tee <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/render_selected_csv.log
```

- [ ] **Step 2: Verify media integrity**

```bash
ffprobe -v error -show_entries format=duration -show_entries stream=codec_name,width,height,avg_frame_rate -of default=noprint_wrappers=1 <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/videos/celebration_t800_armreg_selected_csv.mp4
```

Expected: H.264 video, 640×480, 30 fps, about 10 seconds.

- [ ] **Step 3: Download and visually inspect dense frames**

Download the video, generate a dense contact sheet, and inspect for celebration semantics, arm amplitude, small-arm reversal, lock-up, forward lean, floating feet, and discontinuities. If the video fails, stop before NPZ conversion.

---

### Task 7: Convert the selected CSV to NPZ and verify reconstruction

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/npz/celebration_dance1_subject2_f2880_3180_armreg.npz`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/csv_to_npz.log`

- [ ] **Step 1: Recheck task a and GPU memory**

Run the three checks from Task 4 Step 1. Continue only if task a remains alive and the single-environment converter fits without stopping it.

- [ ] **Step 2: Convert to 50 fps NPZ**

```bash
cd <SERVER_USER_DIR>/engineai_rl_lab-main
TMPDIR=<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/tmp timeout --signal=INT --kill-after=10s 90s \
  <SERVER_USER_DIR>/isaac_env/bin/python scripts/csv_to_npz.py \
  --robot t800 \
  --input_file <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/selected/celebration_dance1_subject2_f2880_3180.csv \
  --input_fps 30 \
  --output_name <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/npz/celebration_dance1_subject2_f2880_3180_armreg.npz \
  --output_fps 50 --headless \
  2>&1 | tee <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/csv_to_npz.log
```

- [ ] **Step 3: Verify NPZ structure and numerical consistency**

Expected:

```text
fps = 50
frames = 499
joint_pos shape = (499, 25)
body_pos_w/body_quat_w/body velocities present
all arrays finite
joint_names exactly match T800 asset order
reconstructed 50 fps CSV agrees with joint_pos/root state within interpolation precision
```

- [ ] **Step 4: Run a 16-environment, 2-iteration smoke test**

```bash
cd <SERVER_USER_DIR>/engineai_rl_lab-main
<SERVER_USER_DIR>/isaac_env/bin/python scripts/tracking/train.py \
  --task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless \
  --num_envs 16 --max_iterations 2 \
  --motion_file <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/npz/celebration_dance1_subject2_f2880_3180_armreg.npz \
  --logger tensorboard --experiment_name task_b_t800_celebration \
  --run_name celebration_armreg_v1_smoke16_iter2 \
  2>&1 | tee <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/train_smoke16_iter2.log
```

Expected: model, `env.yaml`, `agent.yaml`, TensorBoard event file, two completed iterations, and no traceback/OOM/runtime error.

---

### Task 8: Run and evaluate the like-for-like PPO probe

**Files:**
- Create: EngineAI run `*_celebration_armreg_v1_probe256_iter500`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/train_probe256_iter500.log`
- Create: `<SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/videos/celebration_armreg_probe256_iter500_model499.mp4`
- Modify: `<LOCAL_WORKSPACE>\T800_CONTEXT.md`

- [ ] **Step 1: Start one isolated probe only after smoke and memory checks pass**

Use tmux name `task-b-celebration-armreg-probe500`; do not reuse the old probe name. Run default PPO with exactly 256 environments and 500 iterations against only the selected celebration NPZ. Verify tmux, process, first iteration, and GPU memory once after launch.

- [ ] **Step 2: Verify completion artifacts and errors**

After natural completion, require:

```text
model_499.pt
params/env.yaml
params/agent.yaml
events.out.tfevents.*
no Traceback
no CUDA OOM
no RuntimeError
```

- [ ] **Step 3: Compare old and new TensorBoard metrics**

Use the TensorBoard event accumulator to compare:

```text
Mean reward: iteration 0, 499, first-50 average, last-50 average
Mean episode length: same four summaries
Episode_Termination/ee_body_pos: same four summaries
Episode_Reward/action_rate_l2
Metrics/motion/error_joint_pos
Metrics/motion/error_body_pos
sampling_top1_bin/probability/entropy
```

Baseline values to include are old last-50 Mean reward `0.1675`, Mean episode length `10.84`, and `ee_body_pos` `0.999079`.

- [ ] **Step 4: Generate a single-environment checkpoint video**

```bash
cd <SERVER_USER_DIR>/engineai_rl_lab-main
armreg_run_dir=$(find <SERVER_USER_DIR>/engineai_rl_lab-main/logs/rsl_rl/tracking_t800 -mindepth 1 -maxdepth 1 -type d -name '*_celebration_armreg_v1_probe256_iter500' -printf '%f\n' | sort | tail -1)
test -n "$armreg_run_dir"
<SERVER_USER_DIR>/isaac_env/bin/python scripts/tracking/play.py \
  --task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless --num_envs 1 \
  --video --video_length 550 \
  --load_run "$armreg_run_dir" --checkpoint model_499.pt \
  --motion_file <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/npz/celebration_dance1_subject2_f2880_3180_armreg.npz \
  2>&1 | tee <SERVER_USER_DIR>/task_b_t800_actions/armreg_v1/logs/play_probe_model499.log
```

- [ ] **Step 5: Apply the probe video gate**

The video must show continuous recognizable celebration without high-frequency resets, falls, lock-up, strong forward lean, floating feet, or reversed forearms. A numerical improvement without recognizable celebration does not pass.

- [ ] **Step 6: Update the context document and stop at the formal-training decision**

Record source hashes, chosen strength, all candidate metrics, selected CSV/NPZ/video paths, smoke/probe run paths, TensorBoard comparisons, task a status, and the pass/fail decision in `T800_CONTEXT.md`.

Do not start 4096×20000 automatically. Report the evidence and await the user's next instruction. Task a must remain untouched throughout.
