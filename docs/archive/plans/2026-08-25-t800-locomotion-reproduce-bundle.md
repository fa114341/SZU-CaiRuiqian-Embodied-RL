<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 12. Do not use as a complete reproduction guide. -->

# T800 Locomotion Reproduce Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a privacy-safe, self-contained ZIP for the confirmed 12-segment T800 locomotion `model_29999.pt`, including source, assets, data, fixed-speed telemetry, training evidence, reproduction scripts, manifests, and per-file hashes.

**Architecture:** Build into a new local staging directory without modifying the historical server run. Small tested Python utilities sanitize text, inventory files, generate manifests/hashes, scan secrets, export TensorBoard curves, and verify a freshly extracted bundle. A measurement-only SMP evaluator adds left/right foot ContactSensors, forces the steering command to 0.75 m/s, loads the final policy and prior, records per-step telemetry, and emits JSON/NPZ/video without changing the trained policy.

**Tech Stack:** Python 3.13, PyTorch, MuJoCo, MjLab, RSL-RL, NumPy, TensorBoard EventAccumulator, OpenCV, pytest, uv, Docker/NVIDIA Container Toolkit, PowerShell/SSH/SCP for artifact transfer.

---

## File map

Development files created outside the final bundle:

- `task_a_reproduce_bundle_work/bundle_tools.py`: hashing, path sanitization, manifest creation, archive verification, secret scanning.
- `task_a_reproduce_bundle_work/export_curves.py`: TensorBoard scalar export and PNG rendering.
- `task_a_reproduce_bundle_work/build_joint_order.py`: generate joint/action/EE order from the selected source and compiled T800 model.
- `task_a_reproduce_bundle_work/tests/test_bundle_tools.py`: tests for sanitizer, manifest coverage, hashes, and secret scan.
- `task_a_reproduce_bundle_work/tests/test_export_curves.py`: synthetic TensorBoard export test.
- `task_a_reproduce_bundle_work/tests/test_eval_config.py`: test measurement-only ContactSensor and fixed command configuration.

Files added to the selected SMP source snapshot and included in the bundle:

- `source/smp/scripts/evaluate_t800_vx075.py`: deterministic evaluation and telemetry.
- `source/smp/tests/test_t800_vx075_eval.py`: config and output-contract tests.

Generated final bundle files are exactly those defined in the approved design spec.

### Task 1: Freeze inventory and provenance evidence

**Files:**
- Create: `task_a_reproduce_bundle_work/inventory/source_candidates.json`
- Create: `task_a_reproduce_bundle_work/inventory/artifact_inventory.json`
- Create: `task_a_reproduce_bundle_work/inventory/build_notes.md`

- [ ] **Step 1: Confirm the selected immutable inputs**

Read-only check these exact server paths and record size, mtime, and SHA256:

```text
<SERVER_USER_DIR>/smp/logs/tilt_penalty_runs/t800_oldgmr_tilt_w05_free30_ppo30000/2026-08-20_13-07-02_t800_oldgmr_tilt_w05_free30_ppo30000/model_29999.pt
<SERVER_USER_DIR>/smp/logs/pretrain/t800_loco_engineai_official_walk12_fullsmooth31_guard85/20260818_140846/pretrained.pt
<SERVER_USER_DIR>/smp/datasets/norm_stats_t800_engineai_official_walk12_fullsmooth31_guard85.npz
<SERVER_USER_DIR>/smp/datasets/csv/t800_engineai_official_walk12_fullsmooth31_guard85
<SERVER_USER_DIR>/smp/datasets/npz/t800_engineai_official_walk12_fullsmooth31_guard85
```

Expected: one non-empty final checkpoint, one non-empty prior, one norm NPZ, 12 CSV files, and 12 NPZ files. Reject any path containing `walk10` or `drop2`.

- [ ] **Step 2: Compare preserved source candidates**

Compare, file-by-file, excluding caches/logs/data:

```text
server: <SERVER_USER_DIR>/smp/{src,scripts,tests,pyproject.toml,uv.lock,README.md,.python-version}
local:  <LOCAL_WORKSPACE>/smp-master/smp-master/{src,scripts,tests,pyproject.toml,uv.lock,README.md,.python-version}
```

Write JSON entries with relative path, candidate SHA256, mtime, equality, and selected source. Prefer the preserved version at or before the final run time for files that differ; use run `env.yaml`/`agent.yaml` to resolve configuration ambiguity. Record uncertainty rather than claiming byte-exact provenance.

- [ ] **Step 3: Inventory training evidence**

Record the final run's `env.yaml`, `agent.yaml`, TensorBoard event, any original `train.log`, final video, and local confirmed video/contact sheet. Expected: TensorBoard exists; text log may be absent and must be recorded as absent rather than synthesized.

### Task 2: Build tested bundle integrity utilities

**Files:**
- Create: `task_a_reproduce_bundle_work/tests/test_bundle_tools.py`
- Create: `task_a_reproduce_bundle_work/bundle_tools.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
import json

import pytest

from bundle_tools import (
    build_manifest,
    sanitize_text,
    scan_secrets,
    verify_hash_file,
    write_sha256s,
)


[REDACTED: credential or access instruction]
    raw = "motion=<SERVER_USER_DIR>/smp/datasets/a.npz\nroot=C:\\Users\\name\\Desktop\\pkg"
    clean = sanitize_text(raw)
    assert "<SERVER_USER_DIR>" not in clean
    assert "C:\\Users\\name" not in clean
    assert "${BUNDLE_ROOT}/source/smp/datasets/a.npz" in clean


def test_manifest_and_hashes_cover_every_regular_file(tmp_path: Path):
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.bin").write_bytes(b"b")
    manifest = build_manifest(tmp_path, excluded={"SOURCE_MANIFEST.json", "SHA256SUMS.txt"})
    assert {x["path"] for x in manifest["files"]} == {"a.txt", "nested/b.bin"}
    write_sha256s(tmp_path, tmp_path / "SHA256SUMS.txt", excluded={"SHA256SUMS.txt"})
    assert verify_hash_file(tmp_path, tmp_path / "SHA256SUMS.txt") == []


def test_secret_scan_blocks_ipv4_credentials_and_ssh_material(tmp_path: Path):
    (tmp_path / "safe.txt").write_text("vx=0.75", encoding="utf-8")
    assert scan_secrets(tmp_path) == []
[REDACTED: credential or access instruction]
    findings = scan_secrets(tmp_path)
    assert {x["kind"] for x in findings} >= {"credential_assignment", "public_ipv4"}
    (tmp_path / "id_rsa").write_text("not-a-real-key", encoding="utf-8")
    assert any(x["kind"] == "ssh_material_name" for x in scan_secrets(tmp_path))
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest task_a_reproduce_bundle_work/tests/test_bundle_tools.py -q
```

Expected: import failure because `bundle_tools.py` does not yet exist.

- [ ] **Step 3: Implement the minimal utility API**

Implement:

```python
def sha256_file(path: Path) -> str: ...
def sanitize_text(text: str) -> str: ...
def build_manifest(root: Path, excluded: set[str]) -> dict: ...
def write_sha256s(root: Path, output: Path, excluded: set[str]) -> None: ...
def verify_hash_file(root: Path, hash_file: Path) -> list[str]: ...
def scan_secrets(root: Path) -> list[dict[str, str]]: ...
```

`sanitize_text` rewrites POSIX/Windows user-root paths to `${BUNDLE_ROOT}` while retaining relative suffixes. `scan_secrets` inspects UTF-8-compatible files, skips binary model/data/media, rejects non-loopback IPv4 addresses, credential assignments, private-key markers, and SSH-material filenames. It reports kind and relative path but never prints a matched secret value.

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 2 pytest command. Expected: all tests pass.

### Task 3: Stage immutable artifacts and source

**Files:**
- Create directory: `T800_LOCOMOTION_REPRODUCE_12SEG/`
- Populate: `artifacts/`, `source/smp/`, `configs/`, `training/`, `media/`

- [ ] **Step 1: Create a clean staging directory**

Resolve and verify the exact target is under `<LOCAL_WORKSPACE>/` before creation. If an older staging directory exists, rename it with a timestamp instead of deleting it.

- [ ] **Step 2: Transfer selected binary/data artifacts**

Use the existing authenticated SSH channel; do not write connection credentials into commands, scripts, logs, or the bundle. Copy only the confirmed 12-segment checkpoint, prior, norm, 12 CSVs, 12 NPZs, final run configs/event/video, and available training log.

- [ ] **Step 3: Copy the selected complete source snapshot**

Copy `src`, `scripts`, `tests`, `pyproject.toml`, `uv.lock`, `README.md`, and `.python-version`. Exclude `.venv`, `.git`, `__pycache__`, `.pytest_cache`, logs, downloaded checkpoints, datasets outside the selected 12 segments, and editor/system files.

- [ ] **Step 4: Validate immutable artifacts before modification**

Load checkpoint and prior with `torch.load(..., map_location="cpu", weights_only=False)`, load norm and all motion NPZs with NumPy, assert finite arrays and 12-file name equality across CSV/NPZ stems. Record keys, shapes, dtypes, sizes, and SHA256 in `artifact_inventory.json`.

### Task 4: Add fixed-speed telemetry evaluation with ContactSensors

**Files:**
- Create: `task_a_reproduce_bundle_work/tests/test_eval_config.py`
- Create: `T800_LOCOMOTION_REPRODUCE_12SEG/source/smp/tests/test_t800_vx075_eval.py`
- Create: `T800_LOCOMOTION_REPRODUCE_12SEG/source/smp/scripts/evaluate_t800_vx075.py`

- [ ] **Step 1: Write failing config tests**

Tests must import `build_eval_env_cfg()` from the new script and assert:

```python
cfg = build_eval_env_cfg()
cmd = cfg.commands["steering"]
assert cmd.tar_speed_min == 0.75
assert cmd.tar_speed_max == 0.75
assert cmd.rand_tar_dir is False
assert cmd.rand_face_dir is False
assert "push_robot" not in cfg.events
assert cfg.observations["actor"].enable_corruption is False
sensor_names = {s.name for s in cfg.scene.sensors}
assert {"left_foot_ground", "right_foot_ground"} <= sensor_names
```

The tests also assert left/right sensors use primary body patterns `LINK_ANKLE_ROLL_L` and `LINK_ANKLE_ROLL_R`, secondary terrain, fields `("found", "force")`, `reduce="netforce"`, `track_air_time=True`, and history length equal to decimation.

- [ ] **Step 2: Run tests and verify RED**

Expected: import failure because the evaluator does not exist.

- [ ] **Step 3: Implement deterministic configuration and policy loading**

The evaluator must:

```python
TASK_ID = "Smp-Forward-T800-Walk12-TiltPenalty"
TARGET_VX = 0.75

def build_eval_env_cfg():
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.commands["steering"].tar_speed_min = TARGET_VX
    cfg.commands["steering"].tar_speed_max = TARGET_VX
    cfg.commands["steering"].rand_tar_dir = False
    cfg.commands["steering"].rand_face_dir = False
    cfg.events.pop("push_robot", None)
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="left_foot_ground",
            primary=ContactMatch(mode="body", pattern="LINK_ANKLE_ROLL_L", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True, history_length=cfg.decimation,
        ),
        ContactSensorCfg(
            name="right_foot_ground",
            primary=ContactMatch(mode="body", pattern="LINK_ANKLE_ROLL_R", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
            track_air_time=True, history_length=cfg.decimation,
        ),
    )
    return cfg
```

Before `load_env_cfg`, set `SMP_PRETRAINED_CKPT` from a required CLI prior path. Build `ManagerBasedRlEnv`, wrap with `RslRlVecEnvWrapper`, load `model_29999.pt` with the registered runner and `load_cfg={"actor": True}`, and use the deterministic inference policy.

- [ ] **Step 4: Implement telemetry collection**

For every policy step, append CPU NumPy values for:

```text
time_s, episode_step, target_velocity_b[3], base_velocity_b[3], velocity_error_xy,
base_quat_w[4], base_rpy_rad[3], base_height_m, tilt_rad,
joint_pos[25], joint_vel[25], action[25], joint_limit_margin[25],
left/right_found, left/right_force_w[3], left/right_force_norm,
left/right_foot_vel_xy[2], left/right_slip_speed,
left/right_current_air_time, left/right_last_air_time,
left/right_current_contact_time, left/right_last_contact_time,
terminated, truncated, termination_reason
```

Use `robot.data.body_link_lin_vel_w` and root quaternion transforms for base velocity; resolve ankle-roll body IDs once and use their xy link velocity as foot velocity. Slip is foot xy speed only when that side's contact is true. Read termination term names from the termination manager rather than inferring them from pose.

- [ ] **Step 5: Implement JSON/NPZ/schema/video outputs**

Write `metrics_timeseries.npz` with arrays and embedded name arrays, `metrics_summary.json` with units/count/mean/std/p05/p50/p95/max and episode termination totals, and `contact_sensor_schema.json` with source fields and shapes. Record RGB frames through MjLab's `VideoRecorder`; create the contact sheet with OpenCV from ten evenly spaced frames.

- [ ] **Step 6: Run tests and verify GREEN**

Run the new evaluator contract tests plus the existing SMP test suite. Expected: all pass without changing training task semantics.

### Task 5: Run and validate the 0.75 m/s evaluation

**Files:**
- Generate: `evaluation/metrics_summary.json`
- Generate: `evaluation/metrics_timeseries.npz`
- Generate: `evaluation/contact_sensor_schema.json`
- Generate: `evaluation/evaluation.log`
- Generate: `media/t800_locomotion_model_29999.mp4`
- Generate: `media/t800_locomotion_model_29999_contact_sheet.png`

- [ ] **Step 1: Run a 50-step telemetry smoke**

Run the evaluator with the bundle-relative checkpoint/prior and 50 steps. Expected: process exits 0; all required NPZ keys exist; contact arrays have shape `(50,)` or `(50, 3)`; joint/action arrays have shape `(50, 25)`; video decodes.

- [ ] **Step 2: Run the final fixed-speed evaluation**

Run 1000 policy steps at 50 Hz unless the environment terminates and resets; preserve resets and termination reasons in telemetry. Use deterministic policy and exact command `vx=0.75, vy=0, yaw=0`.

- [ ] **Step 3: Validate physical metrics**

Assert all numeric arrays are finite, timestamps monotonic, force norms non-negative, slip is zero-masked out of contact, contact/air durations non-negative, and left/right support statistics are present. Do not impose a fabricated quality threshold; report the measured values honestly.

- [ ] **Step 4: Visually inspect media**

Inspect the contact sheet and video for continuous walking, falling, foot suspension, forward tilt, self-collision, arm behavior, and resets. Record observations in `metrics_summary.json` under `visual_review`.

### Task 6: Export TensorBoard curves and run evidence

**Files:**
- Create: `task_a_reproduce_bundle_work/tests/test_export_curves.py`
- Create: `task_a_reproduce_bundle_work/export_curves.py`
- Generate: `training/curves.csv`
- Generate: `training/curves.png`
- Generate: `training/run_artifacts.json`

- [ ] **Step 1: Write a synthetic event export test**

Create a temporary `SummaryWriter` event with scalar tags `Train/mean_reward` and `Train/mean_episode_length`, call `export_events`, and assert CSV rows and a non-empty PNG.

- [ ] **Step 2: Run test and verify RED**

Expected: missing module/function.

- [ ] **Step 3: Implement event export**

Use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator`; export every scalar tag with columns `tag,step,wall_time,value`. Plot available reward, episode length, velocity/slip, tilt, and termination tags; omit absent tags and list them in `run_artifacts.json` instead of inventing values.

- [ ] **Step 4: Run test and verify GREEN**

Expected: test passes.

- [ ] **Step 5: Export the real final run**

Copy the original event unchanged, export CSV/PNG, parse the maximum scalar step, and record checkpoint/config/event/video hashes. If no original text training log exists, create no fake `train.log`; write `training_log_status: "not_found"` in `run_artifacts.json` and README.

### Task 7: Generate joint/action/EE mapping and source provenance

**Files:**
- Create: `task_a_reproduce_bundle_work/build_joint_order.py`
- Generate: `configs/joint_order.json`
- Generate: `configs/JOINT_ORDER.md`
- Generate: `configs/source_provenance.json`
- Generate: `source/LOCAL_CHANGES.patch`

- [ ] **Step 1: Compile the bundled T800 asset and resolve order**

Assert compiled model joint names after the free joint are exactly `J00_...` through `J24_...`, and `T800_ACTION_SCALE` insertion order matches them one-to-one. Record action scales and HOME default offsets.

- [ ] **Step 2: Resolve SMP EE order**

Read `EE_BODY_NAMES` from `smp.sampling.feature_to_state` and record exact order:

```text
0 LINK_ANKLE_ROLL_L
1 LINK_ANKLE_ROLL_R
2 LINK_BASE
3 LINK_ELBOW_YAW_L
4 LINK_ELBOW_YAW_R
```

State explicitly that `LINK_HEAD_YAW` is the head-yaw child body and is not an SMP EE in this run, while `LINK_WAIST_YAW` is the viewer/body-COM torso anchor used by the environment. Record `J24_HEAD_YAW` and `J12_TORSO_YAW` separately from body-link names.

- [ ] **Step 3: Cross-check dimensions**

Assert policy output dimension is 25, prior feature joint count is 25, feature dimension equals `3 + 6 + 25 + 5*3 + 3 + 3 = 55`, and compiled actuated joint count is 25.

- [ ] **Step 4: Produce provenance and local-change patch**

Compare the selected source against a pinned public SMP baseline when uniquely resolvable. Include the patch and baseline identity. If exact upstream commit cannot be proven, set `upstream_commit_verified=false` and state the comparison basis; do not invent a commit.

### Task 8: Add reproduction scripts and environment documentation

**Files:**
- Create: `scripts/train_reproduce.sh`
- Create: `scripts/evaluate_vx0p75.sh`
- Create: `scripts/verify_bundle.py`
- Create: `environment/Dockerfile`
- Generate: `environment/pip_freeze.txt`
- Generate: `environment/system_info.json`
- Create: `environment/dependency_notes.md`
- Create: `README_REPRODUCE.md`

- [ ] **Step 1: Implement `train_reproduce.sh`**

The script resolves `BUNDLE_ROOT`, verifies hashes, exports package-relative `PYTHONPATH` and `SMP_PRETRAINED_CKPT`, and invokes:

```bash
uv run python source/smp/scripts/train.py Smp-Forward-T800-Walk12-TiltPenalty \
  --env.scene.num-envs="${NUM_ENVS:-4096}" \
  --agent.max-iterations="${MAX_ITERATIONS:-30000}" \
  --agent.run-name="${RUN_NAME:-t800_oldgmr_tilt_w05_free30_ppo30000_reproduce}" \
  --agent.logger=tensorboard \
  --video=False
```

It must not overwrite bundled evidence.

- [ ] **Step 2: Implement `evaluate_vx0p75.sh`**

The script verifies the bundle and invokes the bundled evaluator with relative checkpoint/prior/output paths and fixed 0.75 m/s; it accepts only output directory and step count overrides, not command-speed overrides.

- [ ] **Step 3: Implement `verify_bundle.py`**

It verifies `SHA256SUMS.txt`, manifest coverage, required artifacts, 12 CSV/NPZ stem equality, NPZ readability, and absence of forbidden secret patterns. Exit non-zero with relative paths only.

- [ ] **Step 4: Capture environment**

Run the task-a virtual environment's `pip freeze`; record Python, PyTorch, CUDA runtime, driver, GPU, MuJoCo, MjLab, RSL-RL, NumPy, and OS versions without hostnames, IPs, usernames, process lists, or environment variables.

- [ ] **Step 5: Write Dockerfile and dependency notes**

Use a CUDA runtime compatible with the captured PyTorch build, install `uv`, copy the bundle, sync `source/smp/uv.lock`, set `MUJOCO_GL=egl`, and default to `scripts/verify_bundle.py`. Document NVIDIA Container Toolkit and driver compatibility. Run `docker build --check` if available; otherwise record `docker_build_status: "not_run_tool_unavailable"`.

- [ ] **Step 6: Write README**

README covers contents, provenance limitation, native setup, Docker setup, smoke training, 30000-round training, fixed evaluation, metrics schema, joint/EE mapping links, expected outputs, and privacy statement. All commands use relative paths.

### Task 9: Manifest, privacy audit, archive, and clean-room verification

**Files:**
- Generate: `SOURCE_MANIFEST.json`
- Generate: `SHA256SUMS.txt`
- Generate: `T800_LOCOMOTION_REPRODUCE_12SEG.zip`

- [ ] **Step 1: Run pre-manifest secret scan**

Scan file names and text content. Reject `.env`, SSH material, shell history, non-loopback IPv4, credential assignments, private-key markers, API-key prefixes, known sensitive literals supplied only at runtime, and machine absolute paths in executable docs/configs. Do not print matched secret contents.

- [ ] **Step 2: Build manifest and hashes**

Generate `SOURCE_MANIFEST.json` for every regular file except itself and `SHA256SUMS.txt`, then generate `SHA256SUMS.txt` for every regular file except itself. Manifest entries contain relative path, bytes, SHA256, category, source kind, and sanitized boolean.

- [ ] **Step 3: Run bundle verifier**

Expected: zero missing files, zero hash mismatches, 12/12 motion stem match, loadable model/prior/norm/NPZ, and zero secret findings.

- [ ] **Step 4: Create ZIP without mutation**

Create `T800_LOCOMOTION_REPRODUCE_12SEG.zip` adjacent to staging. Do not add the ZIP inside itself. Record ZIP size and SHA256 outside the archive.

- [ ] **Step 5: Verify a fresh extraction**

Extract into a new temporary directory, run `scripts/verify_bundle.py`, load the final policy/prior/norm on CPU, parse JSON/NPZ/CSV, decode first/middle/last video frames, and recompute all file hashes. Expected: all pass.

- [ ] **Step 6: Run post-extraction secret scan**

Expected: zero findings for IP, password, Token, SSH material, environment secrets, and machine-dependent executable paths.

- [ ] **Step 7: Final delivery record**

Report the local ZIP link, size, ZIP SHA256, selected model/prior hashes, test results, evaluation duration/termination summary, source-provenance caveat, and any Docker verification limitation. Do not report server connection details.

---

## Plan self-review result

- Every approved design requirement maps to a task above.
- The plan contains no unresolved implementation placeholders.
- The selected artifact is consistently the 12-segment `model_29999.pt`; 10-segment assets are explicitly excluded.
- The measurement ContactSensors use compiled T800 ankle-roll bodies because fixed foot links are merged by MuJoCo.
- `LINK_HEAD_YAW`, `LINK_WAIST_YAW`, `J24_HEAD_YAW`, and `J12_TORSO_YAW` are not conflated.
- Secret scanning never stores or prints the known server password/IP.
