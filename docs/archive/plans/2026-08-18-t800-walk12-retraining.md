<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 12. Do not use as a complete reproduction guide. -->

# T800 Walk12 Complete Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the T800 SMP motion prior and PPO locomotion policy from all 12 walking CSV files, automatically progressing through validated stages without overwriting the previous experiment.

**Architecture:** A small Python validation utility enforces file-count, shape, row-count, and finite-value gates. A server-side Bash runner executes each stage, records JSON status and `.done` markers, resumes after disconnects, injects the new prior through an environment variable, and stops on the first failed gate.

**Tech Stack:** Python 3.13, NumPy, PyTorch, MuJoCo/mjlab, Tyro, Bash, SCP/SSH, TensorBoard, RSL-RL.

---

## File Map

- Create: `smp-master/smp-master/src/smp/rl/prior_config.py` — resolve the T800 prior checkpoint from an environment variable with a baseline fallback.
- Modify: `smp-master/smp-master/src/smp/rl/tasks/steering/forward_env_cfg.py` — use the checkpoint resolver.
- Modify: `smp-master/smp-master/src/smp/rl/tasks/steering/__init__.py` — register an isolated `Smp-Forward-T800-Walk12` task and log namespace.
- Create: `smp-master/smp-master/tests/test_prior_config.py` — test default and environment-overridden prior paths.
- Create: `smp-master/smp-master/scripts/validate_walk12.py` — validate CSV, NPZ, checkpoint, and MP4 artifacts.
- Create: `smp-master/smp-master/tests/test_validate_walk12.py` — test successful and failing validation cases.
- Create: `smp-master/smp-master/scripts/run_walk12_pipeline.sh` — restartable automatic server pipeline.
- Create: `walk数据集/walk12_sha256.txt` — upload manifest for all 12 source files.

The workspace is not a Git repository. Instead of commit steps, every changed source file will be copied to a timestamped backup under `~/smp/walk12_backups/` before server replacement, and local diffs will be preserved in the workspace.

### Task 1: Add an overridable prior checkpoint and isolated task name

**Files:**
- Create: `smp-master/smp-master/tests/test_prior_config.py`
- Create: `smp-master/smp-master/src/smp/rl/prior_config.py`
- Modify: `smp-master/smp-master/src/smp/rl/tasks/steering/forward_env_cfg.py`
- Modify: `smp-master/smp-master/src/smp/rl/tasks/steering/__init__.py`

- [ ] **Step 1: Write the failing checkpoint resolver tests**

```python
from smp.rl.prior_config import DEFAULT_T800_PRIOR, resolve_t800_prior_checkpoint


def test_prior_checkpoint_defaults_to_baseline(monkeypatch):
  monkeypatch.delenv("SMP_PRETRAINED_CKPT", raising=False)
  assert resolve_t800_prior_checkpoint() == DEFAULT_T800_PRIOR


def test_prior_checkpoint_uses_environment_override(monkeypatch):
  monkeypatch.setenv("SMP_PRETRAINED_CKPT", "/tmp/walk12/pretrained.pt")
  assert resolve_t800_prior_checkpoint() == "/tmp/walk12/pretrained.pt"
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run on the server after uploading the test:

```bash
cd ~/smp
.venv/bin/python -m pytest tests/test_prior_config.py -q
```

Expected: collection fails because `smp.rl.prior_config` does not exist.

- [ ] **Step 3: Implement the resolver**

```python
"""Resolve the diffusion-prior checkpoint used by SMP locomotion tasks."""

from __future__ import annotations

import os

DEFAULT_T800_PRIOR = "logs/pretrain/t800_loco_lafan/20260817_044744/pretrained.pt"


def resolve_t800_prior_checkpoint() -> str:
  """Return the explicit Walk12 checkpoint or the immutable baseline path."""
  return os.environ.get("SMP_PRETRAINED_CKPT", DEFAULT_T800_PRIOR)
```

- [ ] **Step 4: Use the resolver in the forward environment**

Import `resolve_t800_prior_checkpoint` and replace the hard-coded event parameter with:

```python
cfg.events["init_smp_state"].params["ckpt_path"] = resolve_t800_prior_checkpoint()
```

- [ ] **Step 5: Register an isolated Walk12 task**

Add a second runner configuration in `steering/__init__.py`:

```python
_walk12_rl = unitree_g1_smp_ppo_runner_cfg()
_walk12_rl.experiment_name = "smp_forward_t800_walk12"
_walk12_rl.run_name = "smp_forward_t800_walk12"

register_mjlab_task(
  task_id="Smp-Forward-T800-Walk12",
  env_cfg=g1_forward_smp_env_cfg(play=False),
  play_env_cfg=g1_forward_smp_env_cfg(play=True),
  rl_cfg=_walk12_rl,
)
```

- [ ] **Step 6: Run the resolver tests**

```bash
cd ~/smp
.venv/bin/python -m pytest tests/test_prior_config.py -q
```

Expected: `2 passed`.

### Task 2: Add data and artifact validation

**Files:**
- Create: `smp-master/smp-master/tests/test_validate_walk12.py`
- Create: `smp-master/smp-master/scripts/validate_walk12.py`

- [ ] **Step 1: Write failing tests for CSV and NPZ validation**

Tests must create temporary fixtures and assert:

```python
from pathlib import Path

import numpy as np
import pytest

from scripts.validate_walk12 import validate_csv_directory, validate_npz_directory


def _write_csv(path: Path, rows: int, columns: int) -> None:
  line = ",".join("0.0" for _ in range(columns))
  path.write_text("\n".join(line for _ in range(rows)) + "\n", encoding="utf-8")


def test_source_csv_validation_accepts_expected_shape(tmp_path):
  for index in range(12):
    _write_csv(tmp_path / f"walk_{index:02d}.csv", rows=2, columns=36)
  summary = validate_csv_directory(tmp_path, 12, 36, 24)
  assert summary["valid"] is True
  assert summary["files"] == 12
  assert summary["rows"] == 24


def test_csv_validation_rejects_wrong_column_count(tmp_path):
  _write_csv(tmp_path / "bad.csv", rows=1, columns=35)
  with pytest.raises(ValueError, match=r"bad\.csv.*row 1.*35.*36"):
    validate_csv_directory(tmp_path, 1, 36, 1)


def test_npz_validation_requires_trailing_10_by_55_shape(tmp_path):
  np.savez(tmp_path / "good.npz", windows=np.zeros((3, 10, 55), dtype=np.float32))
  summary = validate_npz_directory(tmp_path, 1, 10, 55)
  assert summary["windows"] == 3

  (tmp_path / "good.npz").unlink()
  np.savez(tmp_path / "bad.npz", windows=np.zeros((3, 10, 59), dtype=np.float32))
  with pytest.raises(ValueError, match=r"bad\.npz.*\(3, 10, 59\)"):
    validate_npz_directory(tmp_path, 1, 10, 55)


def test_npz_validation_rejects_non_finite_values(tmp_path):
  windows = np.zeros((2, 10, 55), dtype=np.float32)
  windows[0, 0, 0] = np.nan
  np.savez(tmp_path / "nan.npz", windows=windows)
  with pytest.raises(ValueError, match=r"nan\.npz.*non-finite"):
    validate_npz_directory(tmp_path, 1, 10, 55)
```

- [ ] **Step 2: Run the validation tests and verify failure**

```bash
cd ~/smp
.venv/bin/python -m pytest tests/test_validate_walk12.py -q
```

Expected: import fails because `scripts.validate_walk12` does not exist.

- [ ] **Step 3: Implement the validator CLI**

Create `scripts/validate_walk12.py` with this implementation:

```python
"""Validate Walk12 CSV, NPZ, and final artifact gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def validate_csv_directory(
  directory: str | Path,
  expected_files: int,
  expected_columns: int,
  expected_total_rows: int,
) -> dict[str, int | bool | str]:
  root = Path(directory)
  files = sorted(root.glob("*.csv"))
  if len(files) != expected_files:
    raise ValueError(f"{root}: found {len(files)} CSV files, expected {expected_files}")

  total_rows = 0
  for path in files:
    with path.open("r", encoding="utf-8", newline="") as handle:
      for row_number, row in enumerate(csv.reader(handle), start=1):
        if len(row) != expected_columns:
          raise ValueError(
            f"{path.name} row {row_number}: found {len(row)} columns, "
            f"expected {expected_columns}"
          )
        try:
          values = np.asarray(row, dtype=np.float64)
        except ValueError as exc:
          raise ValueError(f"{path.name} row {row_number}: non-numeric value") from exc
        if not np.isfinite(values).all():
          raise ValueError(f"{path.name} row {row_number}: non-finite value")
        total_rows += 1

  if total_rows != expected_total_rows:
    raise ValueError(f"{root}: found {total_rows} rows, expected {expected_total_rows}")
  return {
    "kind": "csv",
    "directory": str(root),
    "files": len(files),
    "rows": total_rows,
    "columns": expected_columns,
    "valid": True,
  }


def validate_npz_directory(
  directory: str | Path,
  expected_files: int,
  window_size: int,
  feature_dim: int,
) -> dict[str, int | bool | str]:
  root = Path(directory)
  files = sorted(root.glob("*.npz"))
  if len(files) != expected_files:
    raise ValueError(f"{root}: found {len(files)} NPZ files, expected {expected_files}")

  total_windows = 0
  for path in files:
    with np.load(path, allow_pickle=False) as data:
      if "windows" not in data:
        raise ValueError(f"{path.name}: missing windows array")
      windows = data["windows"]
    if windows.ndim != 3 or windows.shape[1:] != (window_size, feature_dim):
      raise ValueError(
        f"{path.name}: found shape {windows.shape}, "
        f"expected (N, {window_size}, {feature_dim})"
      )
    if not np.isfinite(windows).all():
      raise ValueError(f"{path.name}: non-finite value")
    total_windows += int(windows.shape[0])

  return {
    "kind": "npz",
    "directory": str(root),
    "files": len(files),
    "windows": total_windows,
    "window_size": window_size,
    "feature_dim": feature_dim,
    "valid": True,
  }


def validate_file(path: str | Path, minimum_bytes: int) -> dict[str, int | bool | str]:
  artifact = Path(path)
  if not artifact.is_file():
    raise ValueError(f"{artifact}: file does not exist")
  size = artifact.stat().st_size
  if size < minimum_bytes:
    raise ValueError(f"{artifact}: {size} bytes, expected at least {minimum_bytes}")
  return {"kind": "file", "path": str(artifact), "bytes": size, "valid": True}


def main() -> None:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="kind", required=True)

  csv_parser = subparsers.add_parser("csv")
  csv_parser.add_argument("--directory", required=True)
  csv_parser.add_argument("--expected-files", type=int, required=True)
  csv_parser.add_argument("--expected-columns", type=int, required=True)
  csv_parser.add_argument("--expected-total-rows", type=int, required=True)

  npz_parser = subparsers.add_parser("npz")
  npz_parser.add_argument("--directory", required=True)
  npz_parser.add_argument("--expected-files", type=int, required=True)
  npz_parser.add_argument("--window-size", type=int, required=True)
  npz_parser.add_argument("--feature-dim", type=int, required=True)

  file_parser = subparsers.add_parser("file")
  file_parser.add_argument("--path", required=True)
  file_parser.add_argument("--minimum-bytes", type=int, required=True)

  args = parser.parse_args()
  if args.kind == "csv":
    result = validate_csv_directory(
      args.directory,
      args.expected_files,
      args.expected_columns,
      args.expected_total_rows,
    )
  elif args.kind == "npz":
    result = validate_npz_directory(
      args.directory,
      args.expected_files,
      args.window_size,
      args.feature_dim,
    )
  else:
    result = validate_file(args.path, args.minimum_bytes)
  print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
  main()
```

The CLI must support these exact invocations:

```bash
.venv/bin/python scripts/validate_walk12.py csv \
  --directory datasets/csv/g1_walk12 --expected-files 12 \
  --expected-columns 36 --expected-total-rows 86871

.venv/bin/python scripts/validate_walk12.py csv \
  --directory datasets/csv/t800_walk12 --expected-files 12 \
  --expected-columns 32 --expected-total-rows 86871

.venv/bin/python scripts/validate_walk12.py npz \
  --directory datasets/npz/t800_walk12 --expected-files 12 \
  --window-size 10 --feature-dim 55

.venv/bin/python scripts/validate_walk12.py file \
  --path /absolute/path/to/pretrained.pt --minimum-bytes 1000000
```

Every success prints one JSON object containing file count, row or window count, dimensions, and `"valid": true`. Every failure raises a non-zero exit with the exact offending path.

- [ ] **Step 4: Run the validation tests**

```bash
cd ~/smp
.venv/bin/python -m pytest tests/test_validate_walk12.py -q
```

Expected: all tests pass.

### Task 3: Build the restartable automatic runner

**Files:**
- Create: `smp-master/smp-master/scripts/run_walk12_pipeline.sh`

- [ ] **Step 1: Add strict shell setup and persistent paths**

The runner starts with:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

cd <SERVER_USER_DIR>/smp

STATE_DIR=<SERVER_USER_DIR>/walk12_pipeline
STATUS_FILE="$STATE_DIR/status.json"
MASTER_LOG=<SERVER_USER_DIR>/walk12_pipeline.log
mkdir -p "$STATE_DIR"
```

- [ ] **Step 2: Add stage state and restart logic**

Add these functions. Validation is represented by explicit stages, so a stage is considered restart-safe only after its following validation stage has its own `.done` marker.

```bash
write_status() {
  local stage=$1 state=$2 exit_code=$3 log=$4 artifact=${5:-}
  local timestamp
  timestamp=$(date --iso-8601=seconds)
  .venv/bin/python -c '
import json
import sys
from pathlib import Path

path, stage, state, exit_code, log, artifact, timestamp = sys.argv[1:]
payload = {
    "stage": stage,
    "state": state,
    "timestamp": timestamp,
    "exit_code": int(exit_code),
    "log": log,
    "artifact": artifact,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
' "$STATUS_FILE" "$stage" "$state" "$exit_code" "$log" "$artifact" "$timestamp"
}

run_stage() {
  local stage=$1
  shift
  local done_file="$STATE_DIR/$stage.done"
  local log="$STATE_DIR/$stage.log"

  if [[ -s "$done_file" ]]; then
    write_status "$stage" skipped 0 "$log" "$(cat "$done_file")"
    return 0
  fi

  write_status "$stage" running 0 "$log" ""
  set +e
  "$@" 2>&1 | tee "$log"
  local command_status=${PIPESTATUS[0]}
  set -e
  if (( command_status != 0 )); then
    write_status "$stage" failed "$command_status" "$log" ""
    return "$command_status"
  fi

  date --iso-8601=seconds > "$done_file"
  write_status "$stage" completed 0 "$log" "$(cat "$done_file")"
}
```

- [ ] **Step 3: Add source validation and retarget stages**

```bash
run_stage validate_source \
  .venv/bin/python scripts/validate_walk12.py csv \
  --directory datasets/csv/g1_walk12 --expected-files 12 \
  --expected-columns 36 --expected-total-rows 86871

run_stage retarget \
  .venv/bin/python scripts/retarget_g1_to_t800.py \
  --input-dir datasets/csv/g1_walk12 \
  --output-dir datasets/csv/t800_walk12

run_stage validate_retarget \
  .venv/bin/python scripts/validate_walk12.py csv \
  --directory datasets/csv/t800_walk12 --expected-files 12 \
  --expected-columns 32 --expected-total-rows 86871
```

- [ ] **Step 4: Add NPZ and statistics stages**

```bash
run_stage convert_npz \
  .venv/bin/python scripts/csv_to_npz.py \
  --input-dir datasets/csv/t800_walk12 \
  --output-dir datasets/npz/t800_walk12 \
  --input-fps 30 --output-fps 50 --window-size 10

run_stage validate_npz \
  .venv/bin/python scripts/validate_walk12.py npz \
  --directory datasets/npz/t800_walk12 --expected-files 12 \
  --window-size 10 --feature-dim 55

run_stage norm_stats \
  .venv/bin/python scripts/compute_norm_stats.py \
  --input-dir datasets/npz/t800_walk12 \
  --output datasets/norm_stats_t800_walk12.npz
```

- [ ] **Step 5: Add diffusion training and checkpoint discovery**

```bash
run_stage pretrain env WANDB_MODE=offline \
  .venv/bin/python scripts/pretrain.py \
  --data-dir datasets/npz/t800_walk12 \
  --norm-stats-file datasets/norm_stats_t800_walk12.npz \
  --num-layers 2 --num-epochs 1000 --batch-size 1024 \
  --name t800_loco_walk12 --no-use-wandb --device cuda:0

PRIOR=$(find logs/pretrain/t800_loco_walk12 -type f -name pretrained.pt \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
test -n "$PRIOR"
.venv/bin/python scripts/validate_walk12.py file \
  --path "$PRIOR" --minimum-bytes 1000000
```

Because the pretrain stage is skipped on resume only when its `.done` marker exists, recomputing this newest checkpoint path is deterministic. The file validator prevents progression if it is missing or empty.

- [ ] **Step 6: Add the PPO smoke test**

```bash
run_stage ppo_smoke env \
  SMP_PRETRAINED_CKPT="$PRIOR" CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
  .venv/bin/python scripts/train.py Smp-Forward-T800-Walk12 \
  --env.scene.num-envs 16 --agent.max-iterations 2 \
  --agent.save-interval 1 --agent.logger tensorboard \
  --agent.upload-model False --video False \
  --log-root logs/smoke_walk12
```

The stage passes only when two iterations complete and a non-empty model checkpoint exists under `logs/smoke_walk12`.

- [ ] **Step 7: Add full PPO training**

```bash
run_stage ppo_full env \
  SMP_PRETRAINED_CKPT="$PRIOR" CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline \
  .venv/bin/python scripts/train.py Smp-Forward-T800-Walk12 \
  --env.scene.num-envs 4096 --agent.max-iterations 30000 \
  --agent.save-interval 500 --agent.logger tensorboard \
  --agent.upload-model False --video False
```

The stage passes only when the log contains `Learning iteration 29999/30000`, the process exits normally, and the newest Walk12 run contains a non-empty `model_29999.pt`.

- [ ] **Step 8: Add playback and artifact collection**

Add a playback function that accepts exit 0 or a timeout after the video has been written:

```bash
playback() {
  local model video play_status
  model=$(find logs/rsl_rl/smp_forward_t800_walk12 -type f -name model_29999.pt \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  test -n "$model"

  set +e
  timeout --signal=TERM 20m env \
    SMP_PRETRAINED_CKPT="$PRIOR" CUDA_VISIBLE_DEVICES=0 \
    MUJOCO_GL=egl WANDB_MODE=offline \
    .venv/bin/python scripts/play.py Smp-Forward-T800-Walk12 \
    --checkpoint-file "$model" --video True \
    --num-envs 1 --video-length 400
  play_status=$?
  set -e

  if (( play_status != 0 && play_status != 124 && play_status != 143 )); then
    return "$play_status"
  fi
  video=$(find logs/rsl_rl/smp_forward_t800_walk12 -type f -name '*.mp4' \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  .venv/bin/python scripts/validate_walk12.py file \
    --path "$video" --minimum-bytes 10000
}

run_stage playback playback
```

Collect artifacts without deleting source copies:

```bash
collect_artifacts() {
  local model video event_file
  model=$(find logs/rsl_rl/smp_forward_t800_walk12 -type f -name model_29999.pt \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  video=$(find logs/rsl_rl/smp_forward_t800_walk12 -type f -name '*.mp4' \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
  event_file=$(find logs/rsl_rl/smp_forward_t800_walk12 -type f \
    -name 'events.out.tfevents*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

  mkdir -p <SERVER_USER_DIR>/walk12_artifacts
  cp --preserve=timestamps "$PRIOR" <SERVER_USER_DIR>/walk12_artifacts/pretrained_walk12.pt
  cp --preserve=timestamps "$model" <SERVER_USER_DIR>/walk12_artifacts/model_29999_walk12.pt
  cp --preserve=timestamps "$video" <SERVER_USER_DIR>/walk12_artifacts/walk12.mp4
  cp --preserve=timestamps "$event_file" <SERVER_USER_DIR>/walk12_artifacts/
  cp --preserve=timestamps "$STATUS_FILE" "$MASTER_LOG" <SERVER_USER_DIR>/walk12_artifacts/
  cp --preserve=timestamps "$STATE_DIR"/*.log <SERVER_USER_DIR>/walk12_artifacts/
}

run_stage collect_artifacts collect_artifacts
```

- [ ] **Step 9: Syntax-check the runner**

```bash
bash -n scripts/run_walk12_pipeline.sh
```

Expected: exit code 0 and no output.

### Task 4: Create the local manifest and upload inputs atomically

**Files:**
- Create: `walk数据集/walk12_sha256.txt`

- [ ] **Step 1: Generate the manifest locally**

Create one SHA256 line for each of the 12 CSV files, sorted by filename. Verify the manifest contains 12 unique entries and no other extension.

- [ ] **Step 2: Create server directories without touching baseline data**

```bash
mkdir -p ~/smp/datasets/csv/g1_walk12 \
  ~/smp/datasets/csv/t800_walk12 \
  ~/smp/datasets/npz/t800_walk12 \
  ~/smp/walk12_backups
```

- [ ] **Step 3: Back up the three source files that will be modified**

Copy `forward_env_cfg.py`, `steering/__init__.py`, and any pre-existing Walk12 script to a timestamped folder under `~/smp/walk12_backups/`.

- [ ] **Step 4: Upload the 12 CSV files, manifest, tests, code changes, validator, and runner**

Use SCP to their exact matching paths under `~/smp`. Do not upload the entire repository or overwrite unrelated server files.

- [ ] **Step 5: Verify uploaded file hashes**

Run `sha256sum -c` from `~/smp/datasets/csv/g1_walk12` using the uploaded manifest. Expected: 12 `OK` lines and exit code 0.

### Task 5: Run preflight checks and launch the automatic pipeline

- [ ] **Step 1: Verify resources**

```bash
df -h <SERVER_USER_DIR>
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

Require at least 20 GB free disk and enough free GPU memory for the known 16.8 GB PPO footprint.

- [ ] **Step 2: Run all unit and syntax tests**

```bash
cd ~/smp
.venv/bin/python -m pytest tests/test_prior_config.py tests/test_validate_walk12.py -q
bash -n scripts/run_walk12_pipeline.sh
```

Expected: all tests pass; Bash exits 0.

- [ ] **Step 3: Launch one persistent server-side runner**

```bash
cd ~/smp
nohup bash scripts/run_walk12_pipeline.sh \
  > ~/walk12_pipeline_master.log 2>&1 &
echo $! > ~/walk12_pipeline.pid
```

- [ ] **Step 4: Verify launch**

```bash
test -s ~/walk12_pipeline.pid
ps -p "$(cat ~/walk12_pipeline.pid)" -o pid,etime,cmd
cat ~/walk12_pipeline/status.json
tail -50 ~/walk12_pipeline_master.log
```

Expected: runner process exists, status JSON names the active stage, and no traceback or shell error appears.

### Task 6: Monitor completion and prepare Feishu artifacts

- [ ] **Step 1: Check stage status without modifying the job**

```bash
cat ~/walk12_pipeline/status.json
tail -40 ~/walk12_pipeline_master.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
```

- [ ] **Step 2: Verify final artifacts after automatic completion**

Require the final prior, `model_29999.pt`, TensorBoard event file, final video, stage-status JSON, and all stage logs.

- [ ] **Step 3: Download artifacts to a new local directory**

Download into `<LOCAL_WORKSPACE>\walk12_results\`; do not replace `t800_lafan_model29999.mp4` or previous results.

- [ ] **Step 4: Compare against the three-file baseline**

Record dataset windows, diffusion validation loss, final reward, episode length, self-collision, base-too-low termination, video stability, stride, and arm swing for both runs.

- [ ] **Step 5: Update the Feishu-ready report**

Add the validated curves, final video, exact commands, artifact paths, and a concise explanation of whether full Walk12 data improved the policy.

## Plan Self-Review

- The 12-file dataset flows through every learned stage; no old prior is reused.
- All new paths contain `walk12`, preventing baseline overwrite.
- Every stage has an objective validation gate.
- Automatic progression stops on failures and resumes from validated `.done` markers.
- The prior path is injected without hard-coding a timestamp into source code.
- The final comparison includes behavior as well as reward.
