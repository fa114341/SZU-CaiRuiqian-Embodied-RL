<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# T800 Two Single-Direction Lateral Policies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and train separate fixed-left and fixed-right T800 AMP side-step policies using the two validated official G1-derived motions.

**Architecture:** Add two single-motion task registrations around a shared direction-aware config helper. Preserve the existing mixed task, run red-green configuration tests, then execute smoke, sequential formal training, and deterministic rollout gates.

**Tech Stack:** Python 3.11, MjLab/MuJoCo-Warp, RSL-RL AMP PPO, pytest, tmux, TensorBoard.

---

### Task 1: Freeze the two motion assets

**Files:**
- Create: `src/assets/motions/t800/amp/LateralOfficialLeft/walk_sideway_left_loop_002__A021_t800.npz`
- Create: `src/assets/motions/t800/amp/LateralOfficialRight/walk_sideway_right_loop_001__A022_t800.npz`

- [ ] Copy only the two grounded, QA-passing NPZs into direction-specific directories.
- [ ] Verify each directory contains exactly one NPZ and record SHA256.
- [ ] Commit the assets.

### Task 2: Add single-direction task contracts with TDD

**Files:**
- Modify: `tests/tasks/test_t800_dodge_cfg.py`
- Modify: `src/tasks/amp_loco/config/t800/env_cfgs.py`
- Modify: `src/tasks/amp_loco/config/t800/rl_cfg.py`
- Modify: `src/tasks/amp_loco/config/t800/__init__.py`

- [ ] Add failing tests asserting separate motion directories, fixed `vy=+0.5/-0.5`, zero `vx/wz`, no standing subset, exact runner prior paths, distinct experiment names, and registered task IDs.
- [ ] Run the focused tests and confirm failure because the new configs do not exist.
- [ ] Implement one shared single-direction helper plus explicit left/right wrappers and registrations.
- [ ] Run the focused and existing T800 tests to green.
- [ ] Commit the config implementation.

### Task 3: Make deterministic evaluation task-selectable with TDD

**Files:**
- Modify: `tests/scripts/test_eval_t800_fixed_lateral.py`
- Modify: `scripts/eval_t800_fixed_lateral.py`

- [ ] Add a failing parser/validation test for selecting only one of the two new task IDs.
- [ ] Add `--task-id` and validate its direction against `--vy`.
- [ ] Run focused tests and the complete relevant regression suite.
- [ ] Commit the evaluator change.

### Task 4: Smoke both tasks

- [ ] Verify the stable locomotion seed checkpoint and SHA256.
- [ ] Run left 16 environments × 2 iterations from the seed.
- [ ] Run right 16 environments × 2 iterations from the same seed.
- [ ] Verify exit status, fatal-log scan, checkpoint, YAML, TensorBoard, exact AMP prior path, and first iterations.

### Task 5: Train and gate the two policies

- [ ] Run left 4096 environments × 1000 iterations, saving every 100 iterations.
- [ ] Verify left completion before starting right.
- [ ] Run right 4096 environments × 1000 iterations with identical PPO settings.
- [ ] Evaluate each final checkpoint for 1000 steps with fixed matching `vy` and default standing reset.
- [ ] Generate original-material videos and metrics JSON.
- [ ] Apply the numeric quality gates from the design and record SHA256.
- [ ] Update `T800_CONTEXT.md` with exact paths, results, and remaining blockers.
