<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# T800 Task C Forward-Lateral Dodge Design

## Goal

Build the Task C obstacle-avoidance flow without retraining locomotion from
scratch.  T800 normally walks forward while facing the original heading.  When
an obstacle becomes relevant, it may decelerate or briefly stop and sidestep
left or right; it must not deliberately reverse or turn around.  It returns to
the forward-walk policy after the obstacle has cleared.

## Evidence and constraints

- The released G1 locomotion prior contains 17 motions covering forward,
  reverse, lateral, arc and turning skills, and its command distribution covers
  all of those directions.
- The current T800 locomotion prior contains two forward clips and is trained
  only with `vx=0.4..1.2`, `vy=0`, `wz=0`.
- The current `model_25000` survives a 12-second `vx=0.5`, near-zero `vy/wz`
  replay, but shows heading drift.  It is useful as a forward foundation, not
  as a lateral dodge policy.
- Existing local and server motion libraries contain released G1 side-step
  CSV/NPZ files but no corresponding human BVH.  New lateral references must be
  sourced from human motion and retargeted directly to T800.
- The running locomotion tmux must not be interrupted by Task C work.

## Considered approaches

### 1. Two-policy walk/dodge controller — selected

Keep the current forward policy as the normal locomotion controller.  Train a
separate forward-lateral dodge policy using forward, left-step, right-step and
standing/deceleration references.  A small state machine switches to dodge on
obstacle approach and back to walk after a clear dwell interval.

This isolates regression risk, matches the original project's deployed
walk/dodge separation, and allows the dodge policy to use a narrower motion and
command distribution.

### 2. One unified forward/lateral policy

Continue training the locomotion policy with lateral data and obstacle
observations.  This is simpler at deployment but risks catastrophic forgetting
of the working forward gait and makes failures harder to localize.

### 3. Reward-only dodge without lateral motion references

Train an obstacle policy from the current forward prior using only collision,
clearance and survival rewards.  This needs no new data but is likely to produce
unnatural shuffling, freezing or falling because the AMP prior has no lateral
support transitions.

## Data design

Create a versioned T800 dodge-prior directory containing only:

- the two currently accepted forward references;
- at least two clean left side-step segments;
- at least two independently captured right side-step segments;
- one or two standing/deceleration segments.

The existing LaFAN1 `walk` files contain independently captured, body-relative
left and right side-step windows, so use those before introducing a second human
skeleton.  BONES-SEED/SOMA remains a fallback if fewer than two clips per side
pass T800 replay QA.  Select short, grounded side-step sequences rather than
jumps.  The required processing path is:

`human BVH/SOMA motion -> skeleton mapping -> GMR -> T800 CSV -> joint-limit,
ground-contact and root-trajectory QA -> T800 AMP NPZ`.

Do not use the released G1 robot CSV as an intermediate retargeting source.

Each accepted clip must pass these checks before training:

- no T800 joint-limit violation;
- no foot floating or persistent penetration;
- no forearm inversion;
- torso pitch remains within the existing accepted locomotion envelope;
- left/right root displacement has the expected sign and useful magnitude;
- heading change remains below 10 degrees over the selected segment;
- replay video is reviewed before inclusion.

## Training design

The first dodge policy is command-conditioned but heading-fixed:

- `lin_vel_x = 0.2..0.6 m/s`;
- `lin_vel_y = -0.5..0.5 m/s`;
- `ang_vel_z = 0`;
- no reverse command;
- short standing/deceleration probability is allowed.

Initialize from the best stable forward checkpoint only if network and
observation shapes match; otherwise start a separate run with the same official
PPO defaults.  Keep the AMP prior restricted to motions matching this command
support.  Use staged validation at smoke, probe and formal checkpoints rather
than waiting for a full run to detect data errors.

The obstacle-policy stage adds the original project's state-oracle observation
first.  Depth perception is introduced only after state-oracle dodging passes,
so locomotion, avoidance reward and perception failures remain separable.

## Runtime switching

Use hysteresis to avoid rapid policy switching:

1. Walk policy is active while no obstacle is inside the trigger region.
2. Enter dodge when time-to-contact and predicted body-path intersection cross
   the configured thresholds for several consecutive frames.
3. Dodge policy chooses lateral direction from free-space/obstacle state and may
   reduce `vx` toward zero.
4. Return to walk only after the obstacle is clear for a dwell interval and the
   robot is upright with both command and heading errors bounded.

The first simulation version uses privileged obstacle state.  The same switching
interface later consumes depth-derived obstacle estimates without changing the
two policies.

## Quality gates

### Motion gate

All selected CSV/NPZ references pass joint, contact, root, heading and visual QA.

### Policy gate

- survives a 20-second no-obstacle rollout;
- maintains forward heading without intentional turning;
- completes left and right sidesteps and recenters;
- avoids both left-biased and right-biased obstacle throws;
- no persistent foot sliding, floating, forearm inversion or torso collapse;
- reports termination, collision, minimum clearance, lateral displacement,
  heading error and recovery time metrics.

### Integration gate

The automatic switch completes walk -> dodge -> walk without reset, falls or
high-frequency switching in repeated state-oracle trials.  Only then proceed to
the depth-observation version.

## Failure handling

- If reference replay fails, fix or reject the motion before any RL run.
- If straight walking regresses only in the dodge policy, keep the walk policy
  unchanged and rebalance the dodge prior/commands.
- If lateral displacement is too small, inspect reference root velocities and
  command coverage before changing PPO rewards.
- If the policy turns while `wz=0`, separate heading-command tracking, root-yaw
  drift and viewer-camera effects with quantitative rollout logs.
- If GPU headroom is insufficient while locomotion training is active, defer the
  formal run; do not stop the locomotion tmux.

## Deliverables

- source-motion manifest with licenses and hashes;
- accepted T800 CSV/NPZ references plus replay videos and QA metrics;
- independent dodge run config, checkpoints, TensorBoard and videos;
- state-oracle walk/dodge switching demo and metrics;
- documentation update describing environments, commands and reproduction steps.
