<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# Collision-Guard 30 mm T800 Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a new T800 tracking policy from the collision-aware 30 mm custom-video retargeted motion without interrupting the existing formal run.

**Architecture:** Keep the old 4096-environment training session alive. Convert the corrected 30 FPS CSV into a separate 50 FPS NPZ, validate the motion numerically, pass a short isolated smoke run, then launch a separately named 4096-environment/20000-iteration default-PPO run and verify its first iteration and resource usage.

**Tech Stack:** SSH, tmux, Isaac Lab, engineai_rl_lab, RSL-RL PPO, NumPy, TensorBoard.

---

### Task 1: Audit current state

- [x] Confirm the old formal-training tmux/process remains alive.
- [x] Confirm the corrected CSV exists and inspect GPU/process usage.

### Task 2: Convert and validate the new motion

- [x] Convert `custom_video_t800_collisionaware30mm.csv` from 30 FPS to a separately named 50 FPS NPZ.
- [x] Validate required arrays, shapes, finite values, quaternion norms, and duration.

### Task 3: Run the smoke gate

- [x] Train 16 environments for 2 iterations with the new NPZ and official default PPO.
- [x] Require a zero exit, checkpoints/config/TensorBoard artifacts, and no fatal errors.

### Task 4: Start formal training

- [x] Recheck GPU capacity without stopping existing jobs.
- [x] Start a separate 4096-environment/20000-iteration tmux run using only the new NPZ.
- [x] Verify tmux, PID, first completed iteration, log directory, fatal-error count, and GPU usage.

### Task 5: Record the handoff

- [x] Append exact new-data lineage, commands, run names, paths, and verified state to `T800_CONTEXT.md` without secrets.
- [x] Perform a final fresh status check and report the launch outcome.
