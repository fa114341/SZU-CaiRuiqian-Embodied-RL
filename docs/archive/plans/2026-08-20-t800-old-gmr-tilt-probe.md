<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 3. Do not use as a complete reproduction guide. -->

# T800 Old GMR Tilt-Penalty 30000-Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run an isolated 30000-iteration T800 PPO job using the old successful GMR prior with a gentler forward-tilt penalty and videos every 10000 iterations.

**Architecture:** Add a dedicated environment-config builder and registered task so the baseline task remains unchanged. Verify the config locally, sync only the touched SMP files, smoke-test on the server, then run and render under unique names.

**Tech Stack:** Python 3.13, mjlab, PyTorch, pytest, RSL-RL, MuJoCo EGL

---

### Task 1: Add the isolated tilt-penalty configuration

**Files:**
- Modify: `smp-master/smp-master/tests/test_t800_forward_config.py`
- Modify: `smp-master/smp-master/src/smp/rl/tasks/steering/forward_env_cfg.py`
- Modify: `smp-master/smp-master/src/smp/rl/tasks/steering/__init__.py`

- [ ] Add a failing test asserting speed `0.5–5.0`, reward weight `-0.5`, free angle `math.radians(30)`, and an unchanged default task.
- [ ] Run the focused test and confirm it fails because the dedicated builder is absent.
- [ ] Implement `t800_forward_smp_env_cfg_with_tilt_penalty()` and register `Smp-Forward-T800-Walk12-TiltPenalty`.
- [ ] Run the focused and full local test suites.

### Task 2: Update migration documentation

**Files:**
- Modify: `T800_CONTEXT.md`

- [ ] Replace stale task-a status with the current directory map, server artifacts, experiment history, data-quality findings, and exact next experiment.
- [ ] Verify all recorded local and server paths exist and remove contradictory status text.

### Task 3: Sync and execute the probe

**Files:**
- Server modify: `<SERVER_USER_DIR>/smp/src/smp/rl/tasks/steering/forward_env_cfg.py`
- Server modify: `<SERVER_USER_DIR>/smp/src/smp/rl/tasks/steering/__init__.py`
- Server modify: `<SERVER_USER_DIR>/smp/tests/test_t800_forward_config.py`

- [ ] Upload the three verified files and run the focused server test.
- [ ] Run a two-iteration smoke test with the old prior.
- [ ] Launch30000 iterations under `t800_oldgmr_tilt_w05_free30_ppo30000` with save interval10000.
- [ ] Verify `model_10000.pt`, `model_20000.pt`, and `model_29999.pt`; render and download one video at each node.
