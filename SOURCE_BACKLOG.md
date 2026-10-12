# 服务器独有成果恢复清单

这是一份历史索引，不是已完成备份清单，也不是当前执行任务。按项目所有者 2026-10-12 的决定，本次不再恢复服务器。路径均相对于旧服务器的用户主目录；这里不保存主机名、IP、账号或凭据。下面的文件未重新读取、校验或上传。

| 优先级 | 项目 | 服务器相对位置 | 恢复时要保留 |
| --- | --- | --- | --- |
| P0 | RoboStriker T800 | `.config/superpowers/worktrees/RoboStriker-Preview/t800-stage1-tracking/` | 本地分支、已改文件、未跟踪 T800 脚本与配置；不要只克隆上游 |
| P0 | UFO T800 | `.config/superpowers/worktrees/UFO-main/t800-bringup/` | T800 配置、资产、训练/评估脚本与 Git 补丁 |
| P0 | CBF/PAC-MAN T800 | `perceptive_cbf_rl-t800-work/` | 本地 Git 分支及工作树、训练配置和数据变换 |
| P0 | GMR T800 | `GMR/` | 关节映射、T800 IK 配置、资产与验证脚本 |
| P0 | 自录素材链路 | `robostriker_t800_custom_punch_20260907/`、`task_b_custom_video_*/` | 私有备份原视频、GVHMR/GMR 中间结果；只发布可公开的机器人回放 |
| P1 | SMP 最终 10 段 | `smp/logs/tilt_penalty_runs/t800_oldgmr_walk10_drop2_tilt_w05_free30_ppo30000/` | `model_29999.pt`、prior、norm、env/agent、TensorBoard、视频与指标 |
| P1 | 任务 B | `engineai_rl_lab-main/logs/rsl_rl/tracking_t800/` | 每个正式 run 的 checkpoint、配置与动作版本 |
| P1 | 任务 C 侧移 | `perceptive_cbf_rl-t800-work/.worktrees/task-c-g1npz-two-lateral/` | `model_1699.pt` 和对应评估证据 |
| P1 | UFO 50M/55M | `ufo_t800_combat_mobility_20260912/runs/` | 完整 `checkpoint/`，不仅是模型，还要优化器和训练状态 |
| P1 | RoboStriker Stage1/2/3 | `t800-stage1-tracking/logs/` | Stage1 `model_3223.pt`、Stage2 `model_1998.pt`、Stage3 `model_1999.pt` 及配套配置 |

如未来另行决定恢复服务器，应先在私有存储下载 P0，再做 SHA-256 与可读取性验证，然后按许可/隐私筛选公开内容。大量历史视频与模型未进入本仓库；不能用此公开仓库代替服务器备份。
