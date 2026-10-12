<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 18. Do not use as a complete reproduction guide. -->

# T800 Old-GMR Walk10 Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After the running 12-clip PPO job finishes successfully, train a controlled 10-clip comparison that removes only the two worst clips while retaining the old GMR mapping and the same mild PPO tilt configuration.

**Architecture:** Treat the completed 12-clip job as the gate. Build an immutable ten-file subset from its old-GMR CSV source, regenerate NPZ/norm statistics/prior, run a two-iteration PPO smoke test, and then launch a uniquely named 30000-iteration PPO job. Never start the follow-up if the current job exits without `model_29999.pt`.

**Tech Stack:** Bash, tmux, Python 3.13, PyTorch, SMP diffusion prior, RSL-RL, MuJoCo

---

### Task 1: Gate on successful completion of the current job

**Files:**
- Read: `<SERVER_USER_DIR>/t800_oldgmr_tilt_w05_free30_ppo30000/train.log`
- Read: `<SERVER_USER_DIR>/smp/logs/tilt_penalty_runs/t800_oldgmr_tilt_w05_free30_ppo30000/2026-08-20_13-07-02_t800_oldgmr_tilt_w05_free30_ppo30000/model_29999.pt`

- [ ] Check that `scripts/train.py` for `t800_oldgmr_tilt_w05_free30_ppo30000` is still running. If so, make no server changes and check again later.
- [ ] When it stops, require a nonempty `model_29999.pt` and confirm the log reached `Learning iteration 29999/30000`.
- [ ] If either success condition is missing, report the failure and do not launch Walk10.

### Task 2: Build and validate the old-GMR ten-file subset

**Files:**
- Read: `<SERVER_USER_DIR>/smp/datasets/csv/t800_engineai_official_walk12_fullsmooth31_guard85/*.csv`
- Create: `<SERVER_USER_DIR>/smp/datasets/csv/t800_engineai_official_walk10_fullsmooth31_guard85_drop2/*.csv`

- [ ] Create the destination only if it does not already exist.
- [ ] Copy exactly these ten source files with timestamps preserved:

```text
walk1_subject1.csv
walk1_subject2.csv
walk1_subject5.csv
walk2_subject1.csv
walk2_subject3.csv
walk2_subject4.csv
walk3_subject1.csv
walk3_subject2.csv
walk3_subject3.csv
walk3_subject4.csv
```

- [ ] Confirm these two files are absent:

```text
walk3_subject5.csv
walk4_subject1.csv
```

- [ ] Run:

```bash
cd <SERVER_USER_DIR>/smp
.venv/bin/python scripts/validate_walk12.py csv \
  --directory datasets/csv/t800_engineai_official_walk10_fullsmooth31_guard85_drop2 \
  --expected-files 10 --expected-columns 32 --expected-total-rows 74554
```

Expected: validation succeeds with 10 files, 32 columns, and 74554 total rows.

- [ ] Run `cmp -s` between every kept file and its Walk12 source. Expected: all ten files are byte-identical.

### Task 3: Regenerate Walk10 model inputs

**Files:**
- Create: `<SERVER_USER_DIR>/smp/datasets/npz/t800_engineai_official_walk10_fullsmooth31_guard85_drop2/*.npz`
[REDACTED: credential or access instruction]
- Create: `<SERVER_USER_DIR>/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/npz.log`
- Create: `<SERVER_USER_DIR>/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/norm.log`

- [ ] Convert CSV to NPZ with the same Walk12 settings:

```bash
cd <SERVER_USER_DIR>/smp
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/csv_to_npz.py \
  --input-dir datasets/csv/t800_engineai_official_walk10_fullsmooth31_guard85_drop2 \
  --output-dir datasets/npz/t800_engineai_official_walk10_fullsmooth31_guard85_drop2 \
  --window-size 10 --stride 1 --input-fps 30 --output-fps 50 --device cuda:0
```

- [ ] Validate the NPZ directory:

```bash
.venv/bin/python scripts/validate_walk12.py npz \
  --directory datasets/npz/t800_engineai_official_walk10_fullsmooth31_guard85_drop2 \
  --expected-files 10 --window-size 10 --feature-dim 55
```

Expected: validation succeeds for all ten files.

- [ ] Compute norm statistics:

```bash
.venv/bin/python scripts/compute_norm_stats.py \
  --input-dir datasets/npz/t800_engineai_official_walk10_fullsmooth31_guard85_drop2 \
[REDACTED: credential or access instruction]
```

Expected: the norm-statistics file exists and is nonempty.

### Task 4: Train and verify the Walk10 SMP prior

**Files:**
- Create: `<SERVER_USER_DIR>/smp/logs/pretrain/t800_loco_engineai_official_walk10_fullsmooth31_guard85_drop2/<timestamp>/pretrained.pt`
- Create: `<SERVER_USER_DIR>/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/pretrain.log`

- [ ] Run 1000 prior epochs with the established Walk12 architecture:

```bash
cd <SERVER_USER_DIR>/smp
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline PYTHONUNBUFFERED=1 \
.venv/bin/python scripts/pretrain.py \
  --data-dir datasets/npz/t800_engineai_official_walk10_fullsmooth31_guard85_drop2 \
[REDACTED: credential or access instruction]
  --num-layers 2 --num-epochs 1000 --batch-size 1024 \
  --name t800_loco_engineai_official_walk10_fullsmooth31_guard85_drop2 \
  --no-use-wandb --device cuda:0
```

- [ ] Require the newest `pretrained.pt` to exist and be nonempty. If not, stop before PPO.

### Task 5: Smoke-test and launch the controlled 30000-iteration PPO job

**Files:**
- Create: `<SERVER_USER_DIR>/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/ppo_smoke.log`
- Create: `<SERVER_USER_DIR>/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/train.log`
- Create: `<SERVER_USER_DIR>/smp/logs/tilt_penalty_runs/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/<timestamp>/model_*.pt`

- [ ] Run a two-iteration smoke test with `Smp-Forward-T800-Walk12-TiltPenalty`, 16 environments, and the new Walk10 prior.
- [ ] Require `Learning iteration 1/2` in `ppo_smoke.log` and verify the saved `env.yaml` still records target speed `0.5–5.0`, tilt weight `-0.5`, and free angle `0.5235987756` radians.
- [ ] Launch the full job in tmux session `t800-oldgmr-walk10-drop2-tilt-w05-free30-pipeline` with:

```text
experiment/run name: t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000
task: Smp-Forward-T800-Walk12-TiltPenalty
environments: 4096
iterations: 30000
save interval: 10000
log root: logs/tilt_penalty_runs
WANDB_MODE: offline
```

- [ ] Verify the full process remains alive after its first iteration and report the run directory, prior path, tmux name, and first ETA.
