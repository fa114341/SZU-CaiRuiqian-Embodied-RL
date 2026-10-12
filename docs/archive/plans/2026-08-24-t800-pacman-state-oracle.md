<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 13. Do not use as a complete reproduction guide. -->

# T800 PAC-MAN State-Oracle Migration Implementation Plan

> **Execution mode:** Run this plan in the current task with `superpowers:executing-plans`. Do not use subagents. Do not start a large formal training run without a new user instruction.

**Goal:** Add a native EngineAI T800 task family to PAC-MAN, validate a T800 AMP locomotion environment, and run a minimal state-oracle dodge smoke test while leaving all Unitree G1 tasks unchanged.

**Architecture:** Work in an isolated server copy, `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work`, initialized as a local Git repository for reversible checkpoints. Add a self-contained T800 MJCF/mesh asset and T800 robot constants. Parameterize the small set of robot-structure assumptions used by locomotion, leap/goto, dodge contact, and CBF body points; keep the existing throw, CBF, MimicKit reward, and PPO implementations unchanged. Use existing direct T800 CSV motions to build a T800 AMP NPZ set; never feed a 29-DoF G1 archive to the 25-DoF T800 task.

**Tech Stack:** Python 3.11, mjlab 1.5.3, MuJoCo/MuJoCo Warp, PyTorch 2.9.1+cu128, RSL-RL 5.4.0, pytest.

---

## Guardrails and acceptance criteria

- Preserve `<SERVER_USER_DIR>/perceptive_cbf_rl-main`; all changes occur in `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work`.
- Preserve all registered `Unitree-G1-*` tasks and confirm their configs still build.
- T800 model compiles with 25 actuated joints, finite joint ranges, named foot collision geoms, a root subtree that covers the complete robot, and reserved head cameras.
- A converted T800 motion archive has `joint_pos.shape[-1] == 25`, finite arrays, monotonic frame timing, and body arrays matching the configured T800 AMP bodies.
- `EngineAI-T800-AMP-Flat` and `EngineAI-T800-AMP-Dodge-MimicKit-Flat` appear in `scripts/list_envs.py`.
- A 2-environment T800 dodge environment can reset, throw a ball, produce a 6-D oracle `ball_state`, step actions, and observe a ball-hit signal without exceptions.
- A 16-environment, 2-iteration training smoke produces a checkpoint, config files, TensorBoard event file, and no `Traceback`, `CUDA OOM`, or `RuntimeError`.
- A short single-environment video is generated and visually checked for falling, lock-up, extreme forward lean, floating feet, reversed forearms, and missing ball interaction.
- Stop after the quality gate and report results. Do not start a long run automatically.

## Task 1: Create an isolated, versioned working copy and establish the G1 baseline

**Files:**
- Create: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/`
- Record: `<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/docs/t800_migration/baseline.txt`

1. Verify there are no running PAC-MAN processes or tmux sessions and record GPU memory.
2. Copy the repository without `.venv`, logs, caches, or generated videos; symlink the original `.venv` into the working copy.
3. Initialize a local Git repository and commit the untouched baseline.
4. Run and save the baseline evidence:

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-t800-work
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/list_envs.py
.venv/bin/python - <<'PY'
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
from src.tasks.amp_loco.config.g1.dodge_env_cfgs import g1_amp_dodge_mimickit_flat_env_cfg
for fn in (g1_amp_flat_env_cfg, g1_amp_dodge_mimickit_flat_env_cfg):
    cfg = fn(play=True)
    assert cfg.scene.entities["robot"] is not None
print("G1 configs build")
PY
```

5. Commit checkpoint: `chore: snapshot PAC-MAN baseline before T800 migration`.

## Task 2: Add a self-contained T800 MuJoCo asset and robot constants

**Files:**
- Create: `src/assets/robots/engineai_t800/__init__.py`
- Create: `src/assets/robots/engineai_t800/t800_constants.py`
- Create: `src/assets/robots/engineai_t800/xmls/serial_t800.xml`
- Create: `src/assets/robots/engineai_t800/xmls/assets/*`
- Modify: `src/assets/robots/__init__.py`
- Test: `tests/assets/test_t800_model.py`

1. Write failing tests that compile the model and assert:
   - 25 hinge joints named `J00_*` through `J24_*` plus the free root;
   - root body `LINK_BASE`, left/right feet `LINK_FOOT_L/R`, head `LINK_HEAD_YAW`;
   - named foot collision geoms and `head_camera`/`head_camera_single` cameras;
   - all 25 joints have finite lower/upper limits and lower < upper;
   - the robot config exposes 25 action scales and actuator coverage is exactly 25 joints.
2. Copy the validated source asset from `<SERVER_USER_DIR>/GMR_engineai_work/assets/engineai_t800/xml/serial_t800.xml` and its meshes into the new self-contained asset directory.
3. Patch only the copied MJCF:
   - give collision geoms stable names ending in `_collision`;
   - name the two foot contact geoms `left_foot_collision` and `right_foot_collision`;
   - add `left_foot`/`right_foot` sites;
   - add fixed `head_camera` and `head_camera_single` cameras to `LINK_HEAD_YAW` for the later depth phase;
   - remove embedded `<actuator>` entries if present because mjlab owns actuator creation.
4. Implement `get_t800_robot_cfg()` with the official EngineAI default pose, soft limit factor 0.9, collision config, effort/velocity/armature/stiffness/damping groups, and the published `T800_ACTION_SCALE`.
5. Run the new test and existing tests; commit `feat: add native T800 MuJoCo robot asset`.

## Task 3: Generalize motion conversion and build a T800 AMP archive

**Files:**
- Modify: `scripts/csv_to_npz.py`
- Create: `scripts/validate_motion_npz.py`
- Test: `tests/scripts/test_csv_to_npz_t800.py`
- Create data: `src/assets/motions/t800/amp/WalkAndStand/*.npz`

1. Write a failing unit test around pure conversion helpers using a small synthetic 32-column CSV. Assert quaternion ordering, interpolation, 25-DoF joint output, finite velocity arrays, and expected frame count.
2. Refactor `scripts/csv_to_npz.py` so robot-specific imports and task identifiers are CLI arguments instead of hard-coded G1 values. Keep G1 defaults for backward compatibility.
3. Add `validate_motion_npz.py` to check required keys, finiteness, joint count, body count, quaternion norms, frame duration, root-height range, and joint-limit violations.
4. Select only direct T800 walking/standing CSVs from `<SERVER_USER_DIR>/smp/datasets/csv/t800_engineai_official_walk10_fullsmooth31_guard85_drop2`. Exclude clips previously identified as severe retargeting outliers. Copy the selected CSVs into a manifest-backed staging directory under `<SERVER_USER_DIR>/t800_pacman_data/`.
5. Convert at the task control frequency using the T800 flat tracking environment, then validate every NPZ. Save a JSON manifest with source path, SHA256, frames, duration, speed range, root pitch range, and limit-violation counts.
6. Run converter tests plus a real-archive load test; commit `feat: add 25-dof T800 AMP motion conversion`.

## Task 4: Add a robot profile and T800 AMP locomotion task

**Files:**
- Create: `src/tasks/amp_loco/config/robot_profile.py`
- Modify: `src/tasks/amp_loco/config/g1/env_cfgs.py`
- Modify: `src/tasks/amp_loco/config/g1/rl_cfg.py`
- Create: `src/tasks/amp_loco/config/t800/__init__.py`
- Create: `src/tasks/amp_loco/config/t800/env_cfgs.py`
- Create: `src/tasks/amp_loco/config/t800/rl_cfg.py`
- Modify: `src/tasks/amp_loco/config/__init__.py`
- Test: `tests/tasks/test_t800_amp_cfg.py`

1. Write failing config tests for T800 robot/body/site/action/motion fields and a regression test that the G1 config body names remain unchanged.
2. Introduce an immutable `AmpRobotProfile` containing robot factory, action scale, root body, anchor body, feet bodies/sites/geoms, full-body subtree root, AMP bodies, end-effectors, viewer body, and default root height.
3. Refactor G1 base config construction to accept the G1 profile while preserving public G1 builder names and values.
4. Create the T800 profile:

```python
T800_AMP_BODIES = (
    "LINK_BASE", "LINK_HIP_ROLL_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R", "LINK_KNEE_PITCH_R", "LINK_ANKLE_ROLL_R", "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_YAW_L", "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_YAW_R", "LINK_WRIST_END_R", "LINK_HEAD_YAW",
)
```

5. Build `t800_amp_flat_env_cfg()` with the T800 motion directory and a T800 runner config derived from the existing stock AMP PPO settings, using experiment name `t800_amp_locomotion`.
6. Register `EngineAI-T800-AMP-Flat`, run config tests and `scripts/list_envs.py`, then instantiate 2 environments and take 10 zero-action steps.
7. Commit `feat: register T800 AMP locomotion task`.

## Task 5: Parameterize dodge structure and add the T800 state-oracle task

**Files:**
- Modify: `src/tasks/amp_loco/config/g1/leap_env_cfgs.py`
- Modify: `src/tasks/amp_loco/config/g1/goto_env_cfgs.py`
- Modify: `src/tasks/amp_loco/config/g1/dodge_env_cfgs.py`
- Modify: `src/tasks/amp_loco/mdp/actions_cbf.py`
- Modify: `src/tasks/amp_loco/mdp/events.py`
- Create: `src/tasks/amp_loco/config/t800/dodge_env_cfgs.py`
- Create: `src/tasks/amp_loco/config/t800/dodge_rl_cfg.py`
- Modify: `src/tasks/amp_loco/config/t800/__init__.py`
- Test: `tests/tasks/test_t800_dodge_cfg.py`
- Test: `tests/mdp/test_t800_dodge_integration.py`

1. Write failing tests asserting T800-specific structural references:
   - ball contact secondary subtree is `LINK_BASE`;
   - CBF body candidates resolve to T800 bodies;
   - feet contact and flight rewards reference T800 feet;
   - `ball_state` concatenates to six values;
   - G1 builders retain their existing body names and motion directories.
2. Change structural hard-codes to explicit parameters/profile fields. Do not alter throw speed, timing, CBF gains, MimicKit reward weights, observation math, or PPO hyperparameters.
3. Make the generic leap/goto/dodge builders accept a base task factory, robot profile, and motion directories. Keep existing G1 wrapper functions as backward-compatible adapters.
4. Add the T800 wrapper using `WalkAndStand` for both AMP discriminator and reset-state initialization. For this first state-oracle proof, keep the existing dodge task reward and ball-state observation, but do not claim the archive contains evasive style.
5. Configure `actions_cbf.py` candidate body names from the action config/profile rather than a G1 global tuple. Replace the special `torso_link` lookup in events with the configured anchor body.
6. Register `EngineAI-T800-AMP-Dodge-MimicKit-Flat` with experiment name `t800_amp_dodge_mimickit` and the existing `AMPOnPolicyRunner`/default PPO.
7. Run unit/config tests, instantiate 2 environments, force a deterministic throw, step until impact or timeout, and assert finite observations/rewards plus an active contact/hit path.
8. Commit `feat: add T800 PAC-MAN state-oracle dodge task`.

## Task 6: Run the end-to-end smoke training

**Files/outputs:**
- Create: `<SERVER_USER_DIR>/t800_pacman_runs/<timestamp>_state_oracle_smoke16_iter2/`
- Update: `docs/t800_migration/runbook.md`

1. Run the complete test suite and task-list/config regression checks.
2. Record GPU memory; then start exactly one foreground or manually named tmux smoke run with 16 environments and 2 iterations. Use TensorBoard/offline logging and a unique run name.
3. Verify the process, first iteration, final checkpoint, `env.yaml`, `agent.yaml`, TensorBoard event, and clean exit.
4. Scan the complete log for `Traceback|CUDA out of memory|RuntimeError|nan|inf` and fail the gate on any unexplained match.
5. Commit the runbook and code state: `test: verify T800 state-oracle dodge smoke`.

## Task 7: Generate a diagnostic video and apply the quality gate

**Outputs:**
- Create: `<SERVER_USER_DIR>/t800_pacman_runs/<run>/videos/t800_state_oracle_smoke.mp4`
- Create: `<SERVER_USER_DIR>/t800_pacman_runs/<run>/quality_gate.md`
- Copy for user viewing: `<LOCAL_WORKSPACE>/task_c_videos/t800_state_oracle_smoke.mp4`
- Modify: `<LOCAL_WORKSPACE>/T800_CONTEXT.md`

1. Render one environment headlessly with `MUJOCO_GL=egl`, fixed seed, visible ball trajectory, and at least three throws.
2. Inspect sampled frames and the full video for: meaningful ball reaction, fall, lock-up, forward lean, foot hovering, forearm reversal, mesh/collision mismatch, and camera framing.
3. Record training smoke evidence and a strict verdict:
   - **Pass:** chain is technically valid; recommend a separate user-approved probe, not a formal run.
   - **Conditional pass:** chain runs but motion quality or evasion skill is weak; identify the next smallest data/reward experiment.
   - **Fail:** structural, numerical, or contact issue; fix and repeat Tasks 5-7.
4. Update `T800_CONTEXT.md` with environment path, task IDs, exact commands, outputs, known limitations, and the next manual action. Do not store secrets in new documentation.
5. Report the clickable local video and evidence to the user. Do not begin long training automatically.

## Final verification commands

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-t800-work
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/list_envs.py | grep -E 'Unitree-G1|EngineAI-T800'
.venv/bin/python scripts/validate_motion_npz.py src/assets/motions/t800/amp/WalkAndStand
git status --short
```

Expected final state: tests pass, both T800 task IDs are listed, the 16-env/2-iteration smoke artifacts exist, the diagnostic video is locally viewable, the original G1 repository remains unchanged, and no long training job is running.
