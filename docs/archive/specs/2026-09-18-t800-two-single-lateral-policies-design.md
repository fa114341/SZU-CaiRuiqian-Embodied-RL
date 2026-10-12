<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# T800 Two Single-Direction Lateral Policies Design

## Goal

Replace the failed bidirectional AMP curriculum with two independent T800 low-level skills: one fixed left side-step policy and one fixed right side-step policy.

## Architecture

Each task owns exactly one official G1-derived T800 NPZ, one fixed lateral command, one AMP discriminator, and one experiment namespace. The left task uses `vy=+0.5 m/s`; the right task uses `vy=-0.5 m/s`. Both keep `vx=0`, `wz=0`, the existing T800 action/joint contract, velocity reward weight `3.0`, `std=0.35`, and no delay mask. The old mixed task remains registered only as a historical control.

Both policies resume from the same stable locomotion seed `model_45000.pt`, so balance ability is shared while the two discriminators adapt independently to opposite motion priors. Training is sequential on one RTX 5090: 16 environments × 2 iterations smoke for each task, then 4096 environments × 1000 iterations for left followed by right.

## Data Flow

1. Copy the already validated grounded NPZs into separate immutable asset directories.
2. Register two environment/runner pairs with fixed command signs and matching single-file AMP/reset priors.
3. Run configuration tests and real smoke training.
4. Train left and right policies independently from the same checkpoint seed.
5. Evaluate each final checkpoint for 1000 steps from deployment-style standing reset.

## Quality Gates

- Training: process exit 0; no Traceback, CUDA OOM, RuntimeError, NaN, or killed process; checkpoint/config/TensorBoard present.
- Rollout: mean and tail lateral speed at least `0.30 m/s` in the commanded direction; absolute mean forward drift at most `0.20 m/s`; heading drift at most `15°`; minimum root height at least `0.65 m`; zero non-timeout terminations.
- Artifacts: original-material video, JSON metrics, config files, checkpoint SHA256.

No ball, CBF, camera, or policy-switch logic is added in this phase.
