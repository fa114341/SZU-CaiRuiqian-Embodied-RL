<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 10. Do not use as a complete reproduction guide. -->

# Custom Video A57 T800 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the completed custom-video T800 retargeting CSV into the EngineAI tracking NPZ format and start an isolated 4096-environment, 20000-iteration official-default PPO run.

**Architecture:** Reuse the verified `GVHMR -> SMPL-X -> GMR -> T800 CSV` output without changing its frames or joint order. Convert it with the existing EngineAI Isaac Lab converter, prove the NPZ can load in a 16-environment two-iteration smoke run, then launch a fresh formal run only if the smoke exits cleanly. No repository source files are modified and no other training process is stopped.

**Tech Stack:** EngineAI RL Lab, Isaac Sim 5.1, Isaac Lab, RSL-RL PPO, PyTorch 2.7 CUDA 12.8, T800 25-DoF tracking task.

---

### Task 1: Prepare the isolated run and audit resources

**Files:**
- Read: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/gmr/custom_video_t800.csv`
- Create: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/{npz,train,tmp,logs}`

- [ ] Create the four isolated output directories.
- [ ] Confirm the CSV is `(302, 32)`, finite, and has normalized root quaternions.
- [ ] Record current GPU memory and the user's active Python/tmux processes; do not stop any process.

### Task 2: Convert the T800 CSV to the official tracking NPZ

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/npz/custom_video_t800_50fps.npz`
- Create: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/logs/csv_to_npz.log`

- [ ] Run the existing converter with `--robot t800 --input_fps 30 --output_fps 50 --headless` using `<SERVER_USER_DIR>/isaac_env/bin/python` and the fixed Isaac Lab environment variables.
- [ ] Require converter exit code 0 and a non-empty NPZ.
- [ ] Verify `fps=50`, 25 joint columns, finite arrays, the T800 joint-name order, and approximately 501 output frames.

### Task 3: Run a two-iteration training smoke

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/logs/train_smoke16_iter2.log`
- Create: an isolated run under `<SERVER_USER_DIR>/engineai_rl_lab-main/logs/rsl_rl/tracking_t800/`

- [ ] Train `Tracking-Flat-T800-Wo-State-Estimation-v0` with the new NPZ, 16 environments, two iterations, TensorBoard logging, and run name `customvideo_a57_smoke16_iter2`.
- [ ] Require iterations 0 and 1, `model_0.pt`, `model_1.pt`, `env.yaml`, `agent.yaml`, TensorBoard events, and no Traceback/CUDA OOM/RuntimeError.

### Task 4: Start the formal default-PPO training

**Files:**
- Create: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/logs/train_formal4096_iter20000.log`
- Create: `<SERVER_USER_DIR>/task_b_custom_video_a57a8c0a_20260825/logs/train_formal4096_iter20000.status`
- Create: a formal run under `<SERVER_USER_DIR>/engineai_rl_lab-main/logs/rsl_rl/tracking_t800/`

- [ ] Recheck free GPU memory after the smoke; if another user-owned task is running, leave it untouched and launch only when enough memory remains.
- [ ] Start tmux `task-b-customvideo-a57-formal4096-iter20000` with official default PPO, 4096 environments, 20000 iterations, TensorBoard logger, and run name `customvideo_a57_formal4096_iter20000_defaultppo`.
- [ ] Verify the tmux session, training process, first completed iteration, absence of startup errors, GPU allocation, and the created run directory.
- [ ] Update `T800_CONTEXT.md` with the NPZ, smoke, formal run, log, tmux, first-iteration, and GPU evidence.
