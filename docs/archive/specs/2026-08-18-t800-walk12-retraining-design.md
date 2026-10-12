<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 5. Do not use as a complete reproduction guide. -->

# T800 Walk12 Complete Retraining Design

## Goal

Use all 12 G1 walking CSV files in `<LOCAL_WORKSPACE>\walk数据集` to rebuild the T800 motion dataset, diffusion prior, and PPO locomotion policy without overwriting the previous three-file experiment.

The new experiment must produce independently traceable data, normalization statistics, checkpoints, logs, training curves, and a final walking video for the Feishu report due on 2026-08-22 at 19:00.

## Current Baseline

- Previous dataset: 3 CSV files and 37,698 motion windows.
- Previous diffusion training: 1,000 epochs; final validation loss approximately 0.086.
- Previous PPO training: 30,000 iterations with 4,096 environments.
- Previous final mean reward: 6.21.
- Previous final policy: `logs/rsl_rl/smp_forward_g1/2026-08-17_05-44-32_smp_forward_g1/model_29999.pt`.
- Previous outputs remain immutable comparison artifacts.

## Input Dataset

The new dataset contains 12 headerless CSV files:

- 86,871 source frames in total.
- 29,664,797 bytes in total.
- 36 comma-separated columns per row.
- Layout: 3 root-position values, 4 root-quaternion values, and 29 G1 joint values.
- Expected source frame rate: 30 FPS.

All files are compatible with the existing `scripts/retarget_g1_to_t800.py` input contract.

## Considered Approaches

### 1. Clean full rebuild — selected

Retarget all source CSV files, regenerate NPZ windows and normalization statistics, train a new diffusion prior from scratch, and train a new PPO policy from scratch.

This costs the most compute but gives the cleanest comparison because every learned component uses the same 12-file distribution.

### 2. Fine-tune the previous diffusion prior

Continue diffusion training from the three-file checkpoint, then retrain PPO. This is faster but retains an uncontrolled bias toward the old subset.

### 3. Reuse the old prior and retrain only PPO

This is fastest but the additional nine files would not affect the SMP motion prior, so it does not satisfy the complete-data retraining objective.

## Data and Training Flow

```text
12 G1 CSV files
  -> G1-to-T800 joint retargeting (29 joints to 25 joints)
  -> 12 T800 CSV files with 32 columns
  -> 30 FPS to 50 FPS interpolation and forward kinematics
  -> NPZ windows shaped (N, 10, 55)
  -> q01/q99 normalization statistics from the new NPZ set
  -> diffusion prior training for 1,000 epochs
  -> small PPO smoke test
  -> PPO training with 4,096 environments for 30,000 iterations
  -> final policy, TensorBoard data, video, and comparison summary
```

## Isolated Artifact Layout

No previous directory or checkpoint will be overwritten.

### Local source

`<LOCAL_WORKSPACE>\walk数据集`

### Server inputs and derived data

- G1 CSV: `~/smp/datasets/csv/g1_walk12/`
- T800 CSV: `~/smp/datasets/csv/t800_walk12/`
- NPZ: `~/smp/datasets/npz/t800_walk12/`
- Normalization stats: `~/smp/datasets/norm_stats_t800_walk12.npz`

### Server training outputs

- Diffusion run name: `t800_loco_walk12`
- Diffusion output: the timestamped subdirectory automatically created under `~/smp/logs/pretrain/t800_loco_walk12/`
- Diffusion log: `~/pretrain_walk12.log`
- PPO log: `~/ppo_train_walk12.log`
- PPO experiment/run names must include `walk12` so they cannot collide with `smp_forward_g1` baseline artifacts.

## Fixed Parameters

### Conversion

- Input FPS: 30.
- Output FPS: 50.
- Window size: 10.
- Feature dimension: 55.
- T800 joints: 25.
- End effectors: 5.

### Diffusion prior

- Training from scratch.
- Epochs: 1,000.
- Batch size: 1,024 unless a measured GPU-memory failure requires reduction.
- Train/validation split: 90/10.
- Seed: 42.
- Weights & Biases: offline or disabled.

### PPO

- Environments: 4,096.
- Maximum iterations: 30,000.
- Save interval: 500.
- GPU: the single GPU visible to user `crq` as `cuda:0`.
- Weights & Biases: offline.

## Validation Gates

The next stage starts only after the current gate passes.

## Automatic Stage Progression

The pipeline is unattended after launch. When a validation gate passes, execution automatically starts the next stage without waiting for additional user confirmation:

```text
upload and hash verification
  -> retarget validation
  -> NPZ and normalization validation
  -> diffusion training and checkpoint validation
  -> PPO smoke test
  -> full PPO training
  -> playback and artifact collection
```

Execution pauses instead of advancing when any of the following occurs:

- A validation command returns a non-zero status.
- An expected file, checkpoint, or log is missing or empty.
- A shape, column count, row count, finite-value, or hash check fails.
- Available disk space or GPU memory is insufficient for the fixed parameters.
- A process exits before its required completion marker.
- Continuing would require changing a fixed parameter, modifying the agreed scope, overwriting an existing artifact, or using administrator privileges.

Each stage writes its command output to a persistent log in `<SERVER_USER_DIR>`. A machine-readable stage-status file records the current stage, start and finish times, exit status, validation result, and paths to produced artifacts. This status provides restartability: after a connection loss, completed validated stages are not rerun, and execution resumes from the first incomplete stage.

### Upload gate

- Exactly 12 CSV files exist on the server.
- Server file sizes and SHA256 hashes match the local files.
- Every sampled row has 36 finite numeric columns.
- Total source rows equal 86,871.

### Retarget gate

- Exactly 12 T800 CSV files are generated.
- Each output row has 32 columns: 3 root position, 4 root quaternion, and 25 T800 joints.
- No NaN or infinity values are present.

### NPZ gate

- Exactly 12 NPZ files are generated.
- Every `windows` array has rank 3 and trailing shape `(10, 55)`.
- All arrays contain finite values.
- The total number of windows is recorded for the report.

### Prior gate

- Normalization arrays have 55 features and finite values.
- Diffusion training reaches epoch 999 and writes a non-empty `pretrained.pt`.
- Final training and validation losses are recorded.
- The checkpoint can be loaded by the SMP initialization path.

### PPO smoke gate

- Run a small configuration before the full job.
- The robot model, 55-dimensional prior, observations, rewards, and actions initialize without shape errors.
- At least two iterations complete and a checkpoint is written.

### Full PPO gate

- The 30,000-iteration job runs in the background with persistent logs under `<SERVER_USER_DIR>`.
- Process state, GPU utilization, reward, terminations, and checkpoint creation are monitored.
- Completion requires `model_29999.pt`, a normal process exit, and no fatal error at the end of the log.

### Playback gate

- The final checkpoint produces a non-empty MP4.
- The video is copied to the local workspace.
- Walking stability, stride length, arm swing, self-collision, and falls are reviewed visually.

## Failure Handling

- Never delete or replace baseline artifacts.
- If upload verification fails, re-transfer only the mismatched files.
- If conversion fails, stop before diffusion training and report the exact file and row.
- If batch size 1,024 causes CUDA out-of-memory, retry the prior with batch size 512 and record the deviation.
- If the smoke test fails, do not start the full PPO job.
- If the full PPO process exits early, preserve the log and latest checkpoint before diagnosing.
- Do not use `sudo`; all work stays in `<SERVER_USER_DIR>`.

## Estimated Runtime

- Upload and validation: minutes.
- Retargeting, interpolation, NPZ generation, and statistics: tens of minutes.
- Diffusion prior: approximately 2 hours based on the previous 37,698-window run.
- PPO: approximately 9.5 hours.
- Playback and artifact collection: less than 30 minutes.
- Expected end-to-end duration: approximately 12 hours, subject to GPU contention and preprocessing throughput.

## Report Deliverables

- Dataset inventory and exact file list.
- G1-to-T800 mapping explanation.
- Full conversion and training commands.
- Diffusion train/validation loss.
- PPO reward and termination curves.
- Final checkpoint path.
- Final walking video.
- Comparison against the three-file baseline.
- Explanation of whether the larger dataset improves stride, arm swing, stability, and reward.

## Success Criteria

The experiment is successful when the full pipeline is reproducible, all final artifacts exist, and the video shows stable locomotion. A final reward at or above 6.21 is desirable but not the sole criterion: a lower reward with clearly more natural motion must be preserved and explained rather than discarded.
