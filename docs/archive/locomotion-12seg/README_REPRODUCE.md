<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# T800 Locomotion（12 段）简化复现包

这是已确认效果较好的 12 段 T800 locomotion 成果包。最终运行名为
`t800_oldgmr_tilt_w05_free30_ppo30000`，任务为
`Smp-Forward-T800-Walk12-TiltPenalty`。

## 已包含

- 最终 PPO checkpoint：`artifacts/checkpoint/model_29999.pt`
- SMP prior：`artifacts/prior/pretrained.pt`
- 归一化统计：`artifacts/normalization/norm_stats.npz`
- 12 份动作 CSV 与 12 份转换后 NPZ
- 实际运行保存的 `env.yaml`、`agent.yaml`、TensorBoard event 与完整训练日志
- T800/SMP 源码、URDF、mesh、依赖锁文件
- 最终视频与已有 contact sheet
- action/joint/EE 顺序与关键 link 映射

按当前“简单搞一下”的范围，本包没有重新跑固定速度评估，也没有补造
ContactSensor、速度、滑移、支撑期或终止指标 JSON/NPZ；这些文件不存在时不作伪造。

## 训练参数

- 环境数：4096
- 轮数：30000（checkpoint 内 `iter=29999`）
- PPO：见 `configs/evidence/agent.yaml`
- 目标速度采样范围：0.5–5.0 m/s
- 前倾惩罚：权重 -0.5，30° 免罚区
- 动作 prior：12 段 `t800_engineai_official_walk12_fullsmooth31_guard85`

## Linux 快速启动

先安装 NVIDIA 驱动、CUDA 兼容环境及 `uv`，然后在本目录执行：

```bash
bash scripts/train_reproduce.sh
```

可覆盖参数：

```bash
NUM_ENVS=1024 MAX_ITERATIONS=1000 RUN_NAME=smoke bash scripts/train_reproduce.sh
```

脚本使用包内相对路径，不依赖原服务器目录。完整依赖锁定在
`source/smp/uv.lock`，环境记录见 `environment/dependency_notes.md`。

## 重要说明

源码目录是从最终项目工作目录导出的完整可运行快照，但原目录不是 Git 仓库，
因此无法证明它与训练启动时逐字节一致，也无法提供可信 commit id。实际运行保存的
YAML、checkpoint、日志和 TensorBoard 原件单独保留作为证据。

历史配置中写有 `J13_SHOULDER_ROLL_L: 0.15`，但真实 J13 是
`J13_SHOULDER_PITCH_L`，左肩 roll 实际为 J14。因此该键没有命中真实关节；
本包保留历史事实，没有偷偷修正模型行为。详见 `configs/JOINT_ORDER.md`。

完整性校验：

```bash
sha256sum -c SHA256SUMS.txt
```
