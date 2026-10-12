<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 23. Do not use as a complete reproduction guide. -->

# Original G1 Baseline Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the upstream G1 locomotion baseline end to end before making any further T800 changes.

**Architecture:** Treat `<SERVER_USER_DIR>/perceptive_cbf_rl-main` as a read-only upstream baseline for the initial checks. Validate its registered task and bundled assets, run a small training probe through the official `train_walk.sh` entry, then exercise the bundled `deploy/ckpts/walk_policy.onnx` in the original deployment simulator and capture a reference video. Keep all new run names and outputs isolated from T800 runs.

**Tech Stack:** Ubuntu 22.04, Python 3.11, `uv`, MJLab/MuJoCo, RSL-RL PPO, AMP motion imitation, ONNX Runtime, tmux, ffmpeg.

---

### Task 1: Preserve existing jobs and validate the upstream installation

**Files:**
- Inspect: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/train_walk.sh`
- Inspect: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/src/tasks/amp_loco/config/g1/__init__.py`
- Inspect: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/src/tasks/amp_loco/config/g1/rl_cfg.py`

- [ ] **Step 1: Record current jobs and GPU state**

Run:

```bash
tmux ls
nvidia-smi
pgrep -af 'scripts/train.py|scripts/play.py|deploy/sim/sim_node.py|deploy/policy/dodge_policy.py'
```

Expected: existing jobs are identified before starting anything; no process is stopped.

- [ ] **Step 2: Verify the original G1 task is registered**

Run:

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-main
PYTHONPATH=. uv run python scripts/list_envs.py | grep 'Unitree-G1-AMP-Flat'
```

Expected: output contains `Unitree-G1-AMP-Flat`.

- [ ] **Step 3: Verify motion and deployment assets**

Run:

```bash
find src/assets/motions/g1/amp -type f -name '*.npz' | sort
test -s deploy/ckpts/walk_policy.onnx
test -s deploy/ckpts/dodge_link_cbf.onnx
```

Expected: 17 locomotion NPZ files and both non-empty ONNX files.

### Task 2: Run the official training entry as an isolated smoke test

**Files:**
- Execute: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/train_walk.sh`
- Produce: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/logs/rsl_rl/g1_baseline_probe256/`

- [ ] **Step 1: Start a 256-environment, 20-iteration probe**

Run:

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-main
tmux new-session -d -s g1-baseline-probe256 "NUM_ENVS=256 EXP_NAME=g1_baseline_probe256 bash train_walk.sh --agent.max-iterations=20 --agent.logger=tensorboard --agent.upload-model=False --video False 2>&1 | tee <SERVER_USER_DIR>/g1_baseline_probe256.log"
```

Expected: tmux session `g1-baseline-probe256` exists and the command uses `Unitree-G1-AMP-Flat`.

- [ ] **Step 2: Verify initialization and first iterations**

Run:

```bash
tmux capture-pane -pt g1-baseline-probe256 -S -200
nvidia-smi
```

Expected: iterations advance; no `Traceback`, `CUDA out of memory`, `RuntimeError`, NaN, or asset-loading failure.

- [ ] **Step 3: Verify probe outputs**

Run:

```bash
grep -E 'Traceback|CUDA out of memory|RuntimeError|nan' <SERVER_USER_DIR>/g1_baseline_probe256.log
find <SERVER_USER_DIR>/perceptive_cbf_rl-main/logs/rsl_rl/g1_baseline_probe256 -type f | sort
```

Expected: the error grep has no output; a model checkpoint, `env.yaml`, `agent.yaml`, and TensorBoard event file exist.

### Task 3: Run the bundled deployment policy and capture a reference video

**Files:**
- Execute: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/deploy/play_sim_walk.sh`
- Use: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/deploy/ckpts/walk_policy.onnx`
- Produce: `<SERVER_USER_DIR>/g1_baseline_reference.mp4`

- [ ] **Step 1: Validate the deployment environment and policy dimensions**

Run:

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-main
uv run --project deploy python -c "import onnxruntime as ort; s=ort.InferenceSession('deploy/ckpts/walk_policy.onnx'); print([(x.name,x.shape) for x in s.get_inputs()]); print([(x.name,x.shape) for x in s.get_outputs()])"
```

Expected: policy input and output shapes are printed without import or model errors.

- [ ] **Step 2: Start the original simulator in an isolated tmux session**

Run:

```bash
cd <SERVER_USER_DIR>/perceptive_cbf_rl-main
tmux new-session -d -s g1-baseline-deploy "TASK=Unitree-G1-AMP-Dodge-Depth-Single-BallOnly-Flat VIEWER=viser bash deploy/play_sim_walk.sh deploy/ckpts/dodge_link_cbf.onnx 2>&1 | tee <SERVER_USER_DIR>/g1_baseline_deploy.log"
```

Expected: policy and simulator processes start, the bundled walk ONNX loads, and the viewer endpoint is reported.

- [ ] **Step 3: Capture and inspect the walk rollout**

Use the simulator's walk mode and a forward command within the source motion range. Capture at least 10 seconds and encode it:

```bash
ffmpeg -y -framerate 30 -i <SERVER_USER_DIR>/g1_baseline_frames/%06d.png -c:v libx264 -pix_fmt yuv420p <SERVER_USER_DIR>/g1_baseline_reference.mp4
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 <SERVER_USER_DIR>/g1_baseline_reference.mp4
```

Expected: the video is at least 10 seconds and shows sustained G1 walking without immediate forward collapse, frozen joints, or floating feet.

### Task 4: Produce the migration delta used for the T800 fix

**Files:**
- Inspect: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/src/tasks/amp_loco/config/g1/rl_cfg.py`
- Inspect: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/src/tasks/amp_loco/config/g1/env_cfg.py`
- Inspect: `<SERVER_USER_DIR>/perceptive_cbf_rl-main/src/tasks/amp_loco/config/t800/`
- Update: `<LOCAL_WORKSPACE>/T800_CONTEXT.md`

- [ ] **Step 1: Compare the reproducibility-critical settings**

Record exact G1 and T800 values for motion count and speed coverage, command ranges, default joint pose, actuator stiffness/damping, action scale, reset height/orientation, termination thresholds, AMP bodies/anchor, observation history, PPO network, and reward coefficients.

Expected: every setting has a G1 value, T800 value, and an explicit classification of `matched`, `robot-specific`, or `unintentionally different`.

- [ ] **Step 2: Gate further T800 work on the baseline result**

If the original G1 policy also falls, fix the environment/deployment runtime before touching T800. If G1 walks and T800 falls, use the comparison table to test T800 default stand and reference tracking before another PPO run.

Expected: no new long T800 training is started merely to diagnose a configuration mismatch.
