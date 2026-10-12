<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 1. Do not use as a complete reproduction guide. -->

# T800 Locomotion Lightweight Bundle Implementation Plan

> **For agentic workers:** Execute inline in an isolated staging directory; do not use subagents unless the user explicitly authorizes them.

**Goal:** Package the confirmed 12-segment T800 `model_29999.pt` and all already-existing reproduction evidence into a privacy-safe ZIP without running a new policy evaluation.

**Architecture:** Copy immutable artifacts and a complete source/assets snapshot into a new local staging directory, add concise relative-path reproduction documentation, then generate a file manifest and SHA256 list. Do not modify or stop historical runs, do not fabricate missing telemetry, and do not include connection details or credentials.

**Tech Stack:** PowerShell, SSH/SCP, tar/ZIP, Python/NumPy/PyTorch for read-only artifact checks, SHA256.

---

### Task 1: Inventory and collect existing files

- [ ] Confirm the selected 12-segment checkpoint, prior, norm, 12 CSVs, 12 NPZs, final env/agent YAML, TensorBoard event, existing final video, and any existing text log/contact sheet.
- [ ] Create `<LOCAL_WORKSPACE>/T800_LOCOMOTION_REPRODUCE_12SEG` as a new staging directory; rename rather than delete any pre-existing directory.
- [ ] Copy complete SMP `src`, `scripts`, `tests`, T800 meshes/URDF, `pyproject.toml`, `uv.lock`, README, and Python version file; exclude virtual environments, caches, Git internals, unrelated datasets/logs, and task b/c.
- [ ] Copy the confirmed binary/data artifacts and existing evidence without changing the originals.

### Task 2: Validate artifacts and document order

- [ ] Load checkpoint/prior on CPU and norm/motion NPZs with NumPy; require non-empty, readable artifacts and 12 matching CSV/NPZ stems.
- [ ] Generate `configs/joint_order.json` and `configs/JOINT_ORDER.md` from the actual 25-key `T800_ACTION_SCALE` order and SMP `EE_BODY_NAMES`.
- [ ] Explicitly distinguish `J24_HEAD_YAW`/`LINK_HEAD_YAW` from `J12_TORSO_YAW`/`LINK_WAIST_YAW`; record action scale, HOME offset, and five EE entries.
- [ ] Copy actual run YAMLs as evidence and add sanitized relative-path copies for reproduction.

### Task 3: Add lightweight reproduction information

- [ ] Add `scripts/train_reproduce.sh` using bundle-relative source, data, norm, prior, task name, 4096 environments, and 30000 iterations; permit smoke overrides without embedding machine paths.
- [ ] Capture `pip freeze`, Python/PyTorch/CUDA/MuJoCo/MjLab/RSL-RL/NumPy versions without hostname, IP, credentials, environment variables, or process data.
- [ ] Write `README_REPRODUCE.md` covering artifact identity, source-provenance limitation, environment setup, smoke/full training, existing video, omitted evaluation telemetry, and expected outputs.
- [ ] If no original text log or contact sheet exists, write `not_found` in `training/run_artifacts.json`; do not generate fake evidence.

### Task 4: Manifest, hashes, and privacy scan

- [ ] Generate `SOURCE_MANIFEST.json` with relative path, bytes, SHA256, category, and source kind for every regular file except manifest/hash outputs.
- [ ] Generate `SHA256SUMS.txt` for every regular file except itself.
- [ ] Scan names and text for non-loopback IPv4, credentials, Tokens, SSH/private-key material, shell history, `.env`, absolute executable paths, and known sensitive literals without printing matched values.
- [ ] Verify no task b/c, 10-segment/drop2 assets, or conversation/context files entered the staging directory.

### Task 5: ZIP and clean extraction check

- [ ] Create `T800_LOCOMOTION_REPRODUCE_12SEG.zip` adjacent to staging.
- [ ] Extract into a fresh temporary directory, recompute all hashes, count 12 CSV/12 NPZ, load checkpoint/prior/norm/NPZs, parse YAML/JSON, and decode first/middle/last frames of the existing video.
- [ ] Run the privacy scan again on the extracted tree.
- [ ] Report ZIP path, size, ZIP SHA256, selected model/prior SHA256, included/omitted evidence, and verification results without server connection details.
