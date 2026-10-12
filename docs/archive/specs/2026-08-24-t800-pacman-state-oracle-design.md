<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 4. Do not use as a complete reproduction guide. -->

# T800 PAC-MAN State-Oracle Migration Design

**Date:** 2026-08-24

**Goal:** Migrate the PAC-MAN `perceptive_cbf_rl` state-oracle dodge task from Unitree G1 to EngineAI T800, prove a minimal T800 CBF+RL simulation path, and leave the original G1 tasks unchanged.

## Scope

This phase ends at a small, reproducible T800 state-oracle training and single-environment video. The policy observes ground-truth ball position and velocity; it does not use a depth camera yet. A later phase will add the 16×9 head-mounted ball-only depth observation after the robot, collision, action, AMP, throw, and CBF layers pass independently.

Hardware deployment, ONNX export, ZED/EfficientTAM integration, DPVO, and real-robot control are outside this phase.

## Current Evidence

- Server repository: `<SERVER_USER_DIR>/perceptive_cbf_rl-main`.
- Environment: Python 3.11.15, `torch 2.9.1+cu128`, `rsl_rl 5.4.0` through mjlab; RTX 5090 is currently idle.
- PAC-MAN currently registers only G1 tasks and assumes a 29-DoF action space, G1 body names, G1 collision geom names, G1 camera names, and G1 AMP motion directories.
- T800 source model `<SERVER_USER_DIR>/GMR_engineai_work/assets/engineai_t800/xml/serial_t800.xml` loads in MuJoCo with a floating base, 25 actuated joints, 31 bodies, and 25 actuators.
- That T800 XML does not yet contain PAC-MAN-compatible named collision geoms or head cameras.
- Existing T800 CSVs contain `7 + 25 = 32` values per frame and therefore match the PAC-MAN CSV convention structurally once the converter stops importing the G1 tracking configuration.

## Approaches Considered

### Selected: Native T800 task family inside PAC-MAN

Add T800 assets, constants, configs, motion conversion, and task registrations alongside G1. Reuse robot-independent PAC-MAN ball, CBF, command, reward, and runner modules.

This keeps the fault boundary clear: T800-specific robot details live under `assets/robots/engineai_t800` and `config/t800`; shared avoidance logic remains shared.

### Rejected: Port the depth task first

This combines robot-model, collision, AMP, CBF, observation-layout, camera, and rendering failures in the first experiment. A failed policy would not reveal which layer is wrong.

### Rejected: Wrap the existing SMP policy as a low-level controller

SMP runs in Isaac Lab while PAC-MAN runs in mjlab/MuJoCo Warp. Bridging action semantics, observation history, simulator state, and checkpoint loading would add a second framework boundary before the native T800 task works.

## Architecture

```text
T800 MJCF + actuator constants + T800 AMP motions
                    |
                    v
       T800 AMP locomotion environment
                    |
      ground-truth ball position/velocity
                    |
                    v
     shared PAC-MAN throw + Link-CBF rewards
                    |
                    v
        shared AMPOnPolicyRunner / PPO
                    |
                    v
      checkpoint + metrics + play video
```

The first state-oracle task will use the existing PAC-MAN observation and CBF implementation but substitute all robot-dependent names, dimensions, initial poses, actuator settings, collision groups, and AMP files through a T800 configuration layer.

## Components

### 1. T800 PAC-MAN MJCF

Create a self-contained T800 asset under the PAC-MAN repository instead of referencing `<SERVER_USER_DIR>/GMR_engineai_work` at runtime. Start from the validated `serial_t800.xml` and its meshes.

The adapted MJCF must:

- preserve the official 25-joint order and physical limits;
- use a named floating root joint and a stable home keyframe;
- expose named collision geoms for pelvis/base, torso, head, thighs, shins, feet, upper arms, forearms, and hands/wrist ends;
- distinguish foot collision geoms so ground contact and foot-slip rewards can resolve them;
- use visual meshes only for rendering and primitive geoms for collision where practical;
- retain valid mass and inertia values;
- reserve stable camera names for the later depth phase, even though this phase does not add camera observations.

### 2. T800 Robot Constants

Add a T800 constants module that provides:

- fresh MuJoCo `MjSpec` construction;
- 25-joint actuator groups with stiffness, damping, torque, armature, and velocity limits derived from the official T800 model/config already used by the project;
- a bent-knee standing initial state inside physical limits;
- collision configurations;
- T800 action scales;
- ordered joint, foot, torso, head, and whole-body AMP link names.

The original G1 constants and exports remain unchanged.

### 3. T800 AMP Motion Conversion

Generalize the existing CSV-to-NPZ utility so robot choice selects the matching mjlab tracking configuration and ordered joints. The converter must reject any T800 row that is not exactly 32 columns and must verify finite values, normalized quaternions, and joint limits before producing an NPZ.

For the first smoke, use existing validated T800 walking/standing material. Do not feed the G1 29-DoF AMP archive into the T800 environment. Evasive T800 AMP motions can be added later by direct human-to-T800 retargeting when their source material is available; they are not required to prove the state-oracle software path.

### 4. T800 Task Configuration

Create T800-specific locomotion, state-oracle dodge, and PPO configuration modules. Register new task IDs without renaming or mutating G1 tasks:

- `EngineAI-T800-AMP-Flat`
- `EngineAI-T800-AMP-Dodge-MimicKit-Flat`

The T800 dodge task reuses shared ball spawning, ballistic throws, contact detection, true ball-state observation, Link-CBF reward terms, termination logic, AMP runner, and PPO machinery. Robot-specific names and height/radius thresholds must be calculated or configured for T800 rather than copied blindly from G1.

### 5. Layered Validation

Validation proceeds in this order:

1. Load and compile the T800 MJCF.
2. Assert 25 actuated joints, expected body/geom names, physical joint limits, and valid default pose.
3. Step one T800 environment without a ball and confirm finite observations/actions.
4. Convert and replay a T800 AMP clip; confirm feet contact the ground and the body is not locked, floating, or inverted.
5. Instantiate the T800 state-oracle dodge task and verify ball spawn, trajectory, body collision, Link-CBF metrics, and episode termination.
6. Run a 16-environment, 2-iteration training smoke.
7. If clean, run a small quality probe with an independently named run.
8. Generate a one-environment video and inspect continuous motion, resets, falls, foot suspension, forward lean, arm reversal, and visible avoidance behavior.

Formal large-environment training is not started automatically. It requires the smoke and quality gate to pass and a separate user instruction.

## Error Handling and Preservation

- Work in a timestamped backup/copy because the uploaded server repository has no `.git` history.
- Never overwrite the G1 XML, G1 configs, G1 task IDs, or existing G1 checkpoints.
- Every generated T800 dataset, log, run, checkpoint, and video receives a T800-specific name.
- A failed layer stops progression at that layer; PPO is not used to hide an invalid model, collision setup, or motion file.
- Logs are written under `<SERVER_USER_DIR>`, not `/tmp`.
- No existing task a/b assets or environments are deleted or modified.

## State-Oracle Quality Gate

The phase passes only when all of the following are true:

- T800 MJCF compiles and steps without NaN, invalid inertia, unresolved mesh, or joint-limit errors.
- Exactly 25 policy joints are resolved in the expected order.
- Collision tests identify ball contact against representative torso, head, arm, leg, and foot geoms.
- The converted AMP file passes finite-value, quaternion, joint-limit, and replay checks.
- The 16-environment training smoke exits normally with checkpoint, environment config, agent config, and TensorBoard event files.
- Logs contain no Traceback, CUDA OOM, RuntimeError, or MuJoCo/Warp fatal error.
- The single-environment video shows a continuous T800 episode rather than frequent resets or frozen poses.
- Ball-free behavior remains stable, and thrown-ball episodes expose nonzero CBF/clearance signals plus observable avoidance attempts.

## Next Phase

After this gate passes, add a T800 head-mounted `head_camera_single`, retain the 16×9 ball-only masked depth format, update actor observation dimensions, and train the depth task while using the state-oracle result as the control baseline.
