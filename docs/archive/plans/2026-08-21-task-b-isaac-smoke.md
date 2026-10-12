<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 14. Do not use as a complete reproduction guide. -->

# T800 Task B Isaac Tracking Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start task b without interrupting the running SMP job by restoring the user-space Isaac runtime and proving the official T800 whole-body-tracking path with a two-iteration smoke run.

**Architecture:** Keep task a and task b isolated by directory, tmux name, and log root. Repair the missing native OpenGL dependency in a user-owned directory, upload the repository's official T800 dance motion, then launch a 16-environment, two-iteration headless training smoke. Do not start full task-b training until the smoke exits successfully and a task-specific motion is selected.

**Tech Stack:** Ubuntu 22.04, Isaac Sim 5.1.0, Isaac Lab 2.3.2, engineai_rl_lab 0.1.0, RSL-RL, RTX 5090.

---

### Task 1: Preserve task-a state and establish task-b isolation

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_isaac_smoke/status.log`
- Create: `<SERVER_USER_DIR>/task_b_isaac_smoke/runtime.env`

- [ ] Record the active task-a tmux session, GPU PID, memory use, and timestamp in `status.log`.
- [ ] Require the task-a tmux session to remain present before and after every task-b launch.
- [ ] Reserve tmux name `task-b-t800-dance-smoke` and log root `<SERVER_USER_DIR>/task_b_isaac_smoke`.

### Task 2: Supply `libGLU.so.1` without sudo

**Files:**
- Create: `<SERVER_USER_DIR>/isaac_runtime_libs/`
- Modify: `<SERVER_USER_DIR>/task_b_isaac_smoke/runtime.env`

- [ ] Run `apt-get download libglu1-mesa` from a temporary download directory owned by `crq`.
- [ ] Extract the downloaded package with `dpkg-deb -x` into `<SERVER_USER_DIR>/isaac_runtime_libs`.
- [ ] Write `LD_LIBRARY_PATH=<SERVER_USER_DIR>/isaac_runtime_libs/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH` to `runtime.env`.
- [ ] Verify `ctypes.CDLL("libGLU.so.1")` succeeds under that environment.

### Task 3: Restore the official T800 dance fixture

**Files:**
- Copy: `<LOCAL_WORKSPACE>/engineai_rl_lab-main/engineai_rl_lab-main/datasets/tracking/t800/dance_t800.csv`
- Copy: `<LOCAL_WORKSPACE>/engineai_rl_lab-main/engineai_rl_lab-main/datasets/tracking/t800/dance_t800.npz`
- Create: `<SERVER_USER_DIR>/engineai_rl_lab-main/datasets/tracking/t800/`

- [ ] Upload both files without changing the source files.
- [ ] Compare local and remote SHA256 checksums.
- [ ] Inspect NPZ keys, shapes, dtype, finite values, and quaternion norms before simulation.

### Task 4: Verify headless Isaac startup

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_isaac_smoke/isaac_startup.log`

- [ ] Launch Isaac Sim headless using `<SERVER_USER_DIR>/isaac_env/bin/python` and the task-b runtime library path.
- [ ] Require a zero exit code and an explicit startup/close marker in the log.
- [ ] Recheck task-a tmux and GPU PID after Isaac closes.

### Task 5: Run the two-iteration T800 tracking smoke

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_isaac_smoke/train.log`
- Create: `<SERVER_USER_DIR>/engineai_rl_lab-main/logs/rsl_rl/tracking_t800/<timestamp>/`

- [ ] Launch in tmux with `--task Tracking-Flat-T800-Wo-State-Estimation-v0 --headless --num_envs 16 --max_iterations 2 --motion_file datasets/tracking/t800/dance_t800.npz`.
- [ ] Do not enable video or cameras during the concurrent smoke.
- [ ] Require two completed learning iterations, a non-empty checkpoint/config directory, and no traceback, CUDA OOM, missing-library error, or task-a termination.
- [ ] Stop at the smoke boundary and report results; do not automatically start the 4096-environment task-b run.
