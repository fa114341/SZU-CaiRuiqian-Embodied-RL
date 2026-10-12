<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 33. Do not use as a complete reproduction guide. -->

# GVHMR RTX 5090 Server Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a reproducible GVHMR inference environment on the SSH server, place all six pretrained assets in the paths expected by GVHMR, and prove that the RTX 5090 can run the pipeline without changing the existing SMP, GMR, or Isaac environments.

**Architecture:** Keep the source at `<SERVER_USER_DIR>/GVHMR-main` and the Python 3.10 virtual environment at `<SERVER_USER_DIR>/gvhmr_env`. Use PyTorch 2.7.0 with CUDA 12.8 because the server GPU is an RTX 5090 (`sm_120`); install the remaining GVHMR dependencies separately instead of using the repository's incompatible PyTorch 2.3/CUDA 12.1 pins. Stage weights under `GVHMR-main/inputs/checkpoints` with the exact filenames expected by Hydra.

**Tech Stack:** Python 3.10, uv, PyTorch 2.7/CUDA 12.8, PyTorch3D, Lightning 2.3, Hydra 1.3, Ultralytics, SMPL/SMPL-X, GVHMR.

---

### Task 1: Verify uploads and unpack the source

**Files:**
- Read: `<SERVER_USER_DIR>/input/*`
- Create: `<SERVER_USER_DIR>/GVHMR-main/`

- [ ] **Step 1: Compare the six uploaded file sizes with the local originals**

[REDACTED: credential or access instruction]

- [ ] **Step 2: Extract the source ZIP without requiring the missing `unzip` utility**

```bash
cd <SERVER_USER_DIR>
/usr/bin/python3 -m zipfile -e <SERVER_USER_DIR>/gvhmr_setup_upload/GVHMR-main.zip <SERVER_USER_DIR>
test -f <SERVER_USER_DIR>/GVHMR-main/tools/demo/demo.py
```

Expected: `demo.py` exists and the source ZIP remains unchanged.

### Task 2: Place pretrained assets in GVHMR's exact directory structure

**Files:**
- Create: `<SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`
- Create: `<SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt`
- Create: `<SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth`
- Create: `<SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/yolo/yolov8x.pt`
- Create: `<SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz`
- Create: `<SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl`

- [ ] **Step 1: Create the checkpoint directories**

```bash
mkdir -p <SERVER_USER_DIR>/GVHMR-main/inputs/checkpoints/{gvhmr,hmr2,vitpose,yolo,body_models/smplx,body_models/smpl}
```

- [ ] **Step 2: Extract or copy each model under its canonical filename**

Use Python `zipfile` to copy the single GVHMR, YOLO, SMPL-X, and neutral SMPL members from their uploaded ZIPs. Use `cp --reflink=auto` for the HMR2 and VitPose files, renaming the uploaded `-001` suffix away.

- [ ] **Step 3: Validate the staged assets**

Require all six paths to be non-empty. Compare SHA-256 for HMR2 and VitPose against their uploaded originals, and compare each extracted file's CRC with its ZIP central-directory CRC.

### Task 3: Create the isolated RTX 5090 environment

**Files:**
- Create: `<SERVER_USER_DIR>/gvhmr_env/`

- [ ] **Step 1: Install uv's complete Python 3.10.20 distribution and create the venv**

```bash
<SERVER_USER_DIR>/.local/bin/uv python install 3.10.20
<SERVER_USER_DIR>/.local/bin/uv venv --python 3.10.20 <SERVER_USER_DIR>/gvhmr_env
test -f <SERVER_USER_DIR>/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/include/python3.10/Python.h
```

Do not use `/usr/bin/python3.10`: this server has the interpreter but not `/usr/include/python3.10/Python.h`, so `cython_bbox` and PyTorch3D cannot build C/CUDA extensions against it.

- [ ] **Step 2: Install the RTX 5090-compatible PyTorch pair**

```bash
<SERVER_USER_DIR>/.local/bin/uv pip install --python <SERVER_USER_DIR>/gvhmr_env/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128'
```

- [ ] **Step 3: Prove CUDA and `sm_120` work before installing the rest**

```bash
<SERVER_USER_DIR>/gvhmr_env/bin/python - <<'PY'
import torch
assert torch.cuda.is_available()
assert "sm_120" in torch.cuda.get_arch_list()
x = torch.ones(8, device="cuda")
assert x.sum().item() == 8
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
```

Expected: PyTorch `2.7.0+cu128`, CUDA `12.8`, and NVIDIA GeForce RTX 5090.

### Task 4: Install GVHMR dependencies without downgrading PyTorch

**Files:**
- Modify through package installation only: `<SERVER_USER_DIR>/gvhmr_env/`
- Install editable package: `<SERVER_USER_DIR>/GVHMR-main/`

- [ ] **Step 1: Install the non-PyTorch requirements**

Install the versions from `requirements.txt` except the three incompatible lines for `torch`, `torchvision`, and the PyTorch3D CUDA 12.1/PyTorch 2.3 wheel. Preserve `numpy==1.23.5`, `lightning==2.3.0`, `hydra-core==1.3`, `ultralytics==8.2.42`, `imageio==2.34.1`, and `av==13.0.0`.

- [ ] **Step 2: Install PyTorch3D from the official current source**

```bash
<SERVER_USER_DIR>/.local/bin/uv pip install --python <SERVER_USER_DIR>/gvhmr_env/bin/python iopath fvcore ninja
FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 \
  <SERVER_USER_DIR>/.local/bin/uv pip install --python <SERVER_USER_DIR>/gvhmr_env/bin/python \
  --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'
```

If no usable CUDA compiler is visible, first install `nvidia-cuda-nvcc-cu12==12.8.61`, locate its `nvcc`, set `CUDA_HOME` to the package prefix, and repeat the same source build. Do not install the repository's old `py310_cu121_pyt230` wheel.

- [ ] **Step 3: Install GVHMR editable and run import validation**

```bash
<SERVER_USER_DIR>/.local/bin/uv pip install --python <SERVER_USER_DIR>/gvhmr_env/bin/python -e <SERVER_USER_DIR>/GVHMR-main
<SERVER_USER_DIR>/gvhmr_env/bin/python - <<'PY'
import torch, torchvision, cv2, hydra, lightning, ultralytics, smplx, pytorch3d
from pytorch3d.transforms import axis_angle_to_matrix
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
assert axis_angle_to_matrix(torch.zeros(1, 3, device="cuda")).is_cuda
print("GVHMR_IMPORT_OK")
PY
```

Expected: `GVHMR_IMPORT_OK` with no import or CUDA errors.

### Task 5: Run model-load and official-sample smoke gates

**Files:**
- Read: `<SERVER_USER_DIR>/GVHMR-main/docs/example_video/tennis.mp4`
- Create: `<SERVER_USER_DIR>/GVHMR-main/outputs/demo/`
- Create: `<SERVER_USER_DIR>/gvhmr_setup.log`

- [ ] **Step 1: Validate all checkpoint loaders**

Load the GVHMR Lightning checkpoint, YOLO weights, VitPose weights, HMR2 checkpoint, SMPL-X neutral model, and neutral SMPL model. Require no missing-file, pickle, ZIP, or tensor deserialization errors.

- [ ] **Step 2: Run the official static-camera demo**

```bash
cd <SERVER_USER_DIR>/GVHMR-main
CUDA_VISIBLE_DEVICES=0 <SERVER_USER_DIR>/gvhmr_env/bin/python tools/demo/demo.py \
  --video docs/example_video/tennis.mp4 -s 2>&1 | tee <SERVER_USER_DIR>/gvhmr_setup.log
```

Expected: preprocessing, HMR4D prediction, in-camera render, global render, and merged MP4 all complete without `Traceback`, CUDA OOM, or `RuntimeError`.

- [ ] **Step 3: Record the installation in project memory**

Update `<LOCAL_WORKSPACE>\T800_CONTEXT.md` with the source path, environment path, six checkpoint paths, tested PyTorch/CUDA versions, smoke-test command, and produced official-sample output paths.
