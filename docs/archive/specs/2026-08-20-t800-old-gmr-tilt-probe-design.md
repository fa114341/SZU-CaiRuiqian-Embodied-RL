<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# T800 旧 GMR + 前倾惩罚探针设计

## 目标

使用生成 `T800_final_policy_model_29999.mp4` 的旧版12段GMR数据与对应SMP prior，只加入温和前倾惩罚，进行30000轮PPO训练。

## 对照原则

- 数据固定为服务器 `datasets/csv/t800_engineai_official_walk12_fullsmooth31_guard85` 对应的 prior。
- prior 固定为 `logs/pretrain/t800_loco_engineai_official_walk12_fullsmooth31_guard85/20260818_140846/pretrained.pt`。
- PPO 保持 SMP 默认参数，目标速度保持 `0.5–5.0 m/s`。
- 唯一新增变量是 `excessive_base_tilt_l2`：温和权重 `-0.5`，自由角 `30°`。该权重是2026-08-19版本 `-2.0` 的四分之一，免罚区由 `15°` 放宽为 `30°`。
- 训练30000轮、4096环境，每10000轮保存并生成独立视频。

## 隔离方式

新增任务 `Smp-Forward-T800-Walk12-TiltPenalty`，不修改默认任务 `Smp-Forward-T800-Walk12`。实验名和运行名统一为 `t800_oldgmr_tilt_w05_free30_ppo30000`，日志放入 `logs/tilt_penalty_runs`。

## 验证

1. 单元测试确认新任务保留默认速度范围，并且只增加权重 `-0.5`、自由角 `30°` 的前倾惩罚。
2. 全量测试通过后同步服务器。
3. 服务器两轮 smoke test 检查任务注册、prior 加载和训练链路。
4. 在 `model_10000.pt`、`model_20000.pt`、`model_29999.pt` 节点检查 `env.yaml` 并分别生成视频。

## 判读

本实验与“新映射 + 默认PPO 1000轮”及“旧GMR + 默认PPO 30000轮”形成对照。节点视频用于判断惩罚是否在保留行走能力的同时缓解前倾。
