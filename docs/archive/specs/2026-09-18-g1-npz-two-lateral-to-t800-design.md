<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 1. Do not use as a complete reproduction guide. -->

# G1 NPZ Two-Lateral Motions to T800 Design

## Goal

Convert only the released G1 motions `walk_sideway_left_loop_002__A021.npz` and
`walk_sideway_right_loop_001__A022.npz` into grounded, joint-limited T800
references. Produce numerical reports and direct-replay videos before any PPO
training.

## Scope and source contract

- Read the two immutable files from the untouched G1 PAC-MAN repository.
- Require the released schema: `fps`, `joint_pos`, `joint_vel`, `body_pos_w`,
  `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`.
- Require 30 G1 bodies in compiled-model order, excluding the MuJoCo world body.
- Treat `body_quat_w` as scalar-first `wxyz`, matching MjLab.
- Do not copy G1 joint angles into T800 joints and do not modify the source NPZs.

## Coordinate and target construction

Normalize each motion into the first pelvis heading frame: subtract the first
pelvis horizontal position and rotate positions and orientations by the inverse
initial pelvis yaw. Preserve the original body-frame lateral root displacement;
the accepted left clip must remain positive lateral and the accepted right clip
negative lateral.

Drive T800 through task-space IK. Track the pelvis/root, bilateral hip/knee/ankle,
torso, shoulder, elbow, and wrist landmarks. Root and feet receive the strongest
costs; knees/hips are positional aids; arm targets are low weight and combined
with the known T800 action-scale-aware neutral-arm posture. For each mapped body,
apply motion deltas relative to source frame zero around the corresponding T800
neutral landmark. Apply source relative rotations to the T800 neutral rotation
only where orientation tracking is enabled. This avoids assuming that G1 and
T800 link coordinate frames or neutral offsets are identical.

## Constraints and post-processing

- Solve with Mink against the EngineAI T800 MuJoCo model.
- Enforce an 85% guard band inside T800 hard joint limits.
- Warm-start frame zero, then solve frames sequentially for temporal continuity.
- Smooth root and joint trajectories without changing motion direction.
- Convert MuJoCo root quaternion `wxyz` to the repository CSV `xyzw` layout.
- Use the existing `csv_to_npz.py --ground-feet` path to place T800 sole collision
  geometry at `-2 mm` and regenerate all T800 FK fields.

## Outputs

Write isolated artifacts under
`<SERVER_USER_DIR>/task_c_t800_dodge/g1_npz_two_lateral_v1/`: source manifest, CSV, NPZ,
per-motion metrics, logs, T800 replay videos, and a two-video contact sheet. No
output is added to a training glob in this phase.

## Acceptance gates

Both motions must be finite, remain inside the 85% joint-limit guard, have
normalized root quaternions, preserve opposite lateral directions, show at least
`0.30 m/s` median/mean commanded-direction root speed, keep absolute forward
drift below `0.20 m/s`, keep root heading drift below `15 degrees`, and replay
without obvious floating, foot penetration, inverted limbs, or discontinuous
jumps. A failed gate stops before training and is reported with the video.
