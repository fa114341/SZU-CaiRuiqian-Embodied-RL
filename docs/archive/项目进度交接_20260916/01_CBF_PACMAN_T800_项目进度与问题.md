<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 22. Do not use as a complete reproduction guide. -->

# CBF / PAC-MAN T800 避障项目：进度、卡点与技术交接

> 更新日期：2026-09-16
> 用途：供后续读者理解 `perceptive_cbf_rl`（PAC-MAN / Link-CBF）迁移到 EngineAI T800 的真实进度并协助分析。
> 安全说明：本文不含服务器 IP、密码、Token 或私钥。

## 1. 一句话结论

T800 的模型、AMP、PPO、投球、真实球状态观测和 Link-CBF 工程链路已经跑通，原版 G1 baseline 也已确认可用；但 T800 至今没有学出可靠的左右侧步避球。当前项目停在“先把独立左右侧步技能练实并做固定速度验收”这一层，尚未进入深度相机阶段，也没有可称为成功的 T800 避障策略。

截至本次整理，服务器上没有 CBF/PAC-MAN 训练进程或 tmux 会话在运行。

## 2. 项目目标与范围

上游项目为 `perceptive_cbf_rl`，任务目标是让人形机器人面对来球时保持稳定，并根据安全方向做侧向躲避。现阶段采用 state-oracle：策略直接读取仿真中的球位置和速度，不使用头部相机图像。

期望控制链路为：

```text
机器人本体历史观测 + 名义运动命令 + 真实球状态
→ PPO/AMP 低层策略
→ 25维 T800 关节目标
→ Link-CBF / 安全命令或安全指标约束
→ 前进、停步或左右侧移躲避
```

最终设计倾向于“双策略 + 滞回切换”：无威胁时使用稳定 locomotion，威胁出现时切换 dodge，威胁持续消失后再切回。但该部署逻辑虽已做过代码原型，前提所需的可靠侧步/避障策略尚未通过行为门禁。

## 3. 代码、环境和机器人合同

### 3.1 仓库与运行环境

```text
原始 G1 工程（保持不动）：<SERVER_USER_DIR>/perceptive_cbf_rl-main
T800 工作副本：<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work
T800 Python：<SERVER_USER_DIR>/perceptive_cbf_rl-t800-work/.venv/bin/python
本地代码镜像：<LOCAL_WORKSPACE>\task_c_pacman_work
```

### 3.2 T800 基本合同

- 原生动作维度：25。
- 根节点：`LINK_BASE`。
- PAC-MAN 躯干/AMP anchor：`LINK_WAIST_YAW`。
- 脚部、手部、IMU、接触和预留相机均已按 T800 资产重新配置。
- 原始投球逻辑、ball-hit termination、AMP runner、PPO 和大部分 Link-CBF 实现保持上游语义。
- T800 新任务：`EngineAI-T800-AMP-Flat`、`EngineAI-T800-AMP-Dodge-MimicKit-Flat`、`EngineAI-T800-AMP-Lateral-Flat`。

## 4. 已经完成并验证的工作

### 4.1 T800 state-oracle 技术链路

16 环境 × 2 轮技术冒烟成功：动作维度25，actor能读取 `ball_state`，投球和碰撞可触发，模型、YAML、TensorBoard、ONNX 均能保存。日志无 Traceback、CUDA OOM 或 RuntimeError。

这只证明“能训练”，不证明“会躲”。随机初始策略在视频中会摔倒。

### 4.2 原版 G1 baseline 复核

原仓库的 G1 训练和部署链路已经独立验证：

- 256 环境 × 20 轮训练冒烟成功。
- 官方 `walk_policy.onnx` 在固定 `vx=0.5 m/s` 下持续行走。
- 平均存活约1560步（31.2秒），验证时某回合超过4000步。
- 官方避障策略文件也存在：`<SERVER_USER_DIR>/perceptive_cbf_rl-main/deploy/ckpts/dodge_link_cbf.onnx`。

因此 T800 失败不能归因于“原 PAC-MAN 工程本来就跑不通”。

### 4.3 Locomotion 数据的两个确定错误及修复

第一，旧参考数据包含严重根俯仰异常：`walk3_subject4` 某段根俯仰接近89.6°。已增加 `max_abs_root_pitch_deg` 数据门禁，并换成绝对根俯仰不超过25°的窗口。

第二，旧 T800 参考动作整体悬空：最低脚底中位数约高于地面50–58 mm。已按实际脚底支撑 geom 逐帧校正 root Z 到约 `-2 mm` 接触，并把 T800 名义 root 高度从 `1.06 m` 改为 `1.0145 m`。

同时把仅有前进行走数据的训练命令从 G1 的全向范围收紧到：

```text
vx = 0.4..1.2 m/s
vy = 0
yaw rate = 0
```

### 4.4 可用但已停止的 locomotion checkpoint

8192 环境正式续训曾运行到 `model_45000.pt`，当时单环境固定前进和零速站立都可以稳定10秒；后续它还成功 sim2sim 到 RoboStriker/Isaac Lab。用户后来明确取消任务 a，因此训练被停止，但 checkpoint、日志和视频都保留。

```text
[REDACTED: credential or access instruction]
最终保留节点：model_45000.pt
本地视频：<LOCAL_WORKSPACE>\task_c_videos\t800_groundedv3_home10145_resume1998.mp4
```

`model_45000.pt` 只证明前进/站立能力，不含可靠横移、后退和转向。

## 5. 避障与侧步实验结果

| 实验 | 规模/起点 | 结果 | 结论 |
| --- | --- | --- | --- |
| state-oracle 技术 smoke | 16 env × 2 iter | 链路和产物正常，随机策略摔倒 | 工程通过，行为失败 |
| 前向+5段侧步 dodge probe | 256 env × 1000 iter | episode length约48→116；有球仍原地调整且被击中 | 未学出侧移 |
| 基线续训 | 从 `model_999`，4096 env 到 `model_2999` | 站立改善，但仍没有清晰侧移 | 单纯加轮数无效 |
| safe-velocity 奖励 A/B | 从同一 `model_2000`，2048 env × 500 | `ball_hit` 与基线几乎相同，视频仍原地 | 安全速度奖励目标不够 |
| 硬 CBF 命令 | `cbf_filter_command=True`，2048 env × 500 | 后段失稳，仍无稳定侧移 | 仅暴露安全命令不能创造未学会的技能 |
| 独立 lateral curriculum | 5段纯侧步，4096 env × 1000 | Mean episode length约983，但固定 `vy≈0.497` 时实际横向速度约0 | 策略学会站立并忽略命令 |
| 加强横向速度跟踪 | 从 `model_45999`，4096 env × 500 | 正常结束；末轮 episode length约985，`error_vel_xy≈0.8345` | 训练完成，但固定左右速度/视频验收未做完 |

最后一轮配置改动为：

```text
rel_standing_envs: 0.05 -> 0.0
track_anchor_linear_velocity weight: 1.0 -> 3.0
velocity reward std: 1.0 -> 0.35
mask_delay: True -> False
commit: 0893ea4 fix: enforce lateral velocity tracking
```

最后一轮日志明确包含 `TRAIN_EXIT_0`，但不能凭高 reward 或长 episode 判定成功；缺失的关键证据是同一 checkpoint 下固定左移和固定右移的实际机体速度、航向保持和视频。

最后一轮可评估的精确产物为：

```text
[REDACTED: credential or access instruction]
checkpoint：上述目录/model_46498.pt
```

## 6. 当前卡点和较可信的根因

### 6.1 CBF 不等于运动技能生成器

CBF 可以给出安全方向、过滤命令或提供安全约束，但若低层策略本身不会稳定侧移，CBF 不会自动生成可执行的侧步。硬 CBF 实验失败是这一点的直接证据。

### 6.2 原 MimicKit 目标对 T800 缺乏明确侧向可辨识性

原 MimicKit 分支有意设置：

- `cbf_filter_command=False`；
- `dodge_sidestep` reward 权重为0；
- 主要靠躯干远离球、碰撞终止、per-link CBF 和 AMP 来形成动作。

这在 G1 上可能依赖成熟的全向动作先验，但 T800 的初始动作库和低层能力不同。前几轮策略发现“站稳不动”是更容易的局部最优。

### 6.3 数据覆盖和命令覆盖曾严重错配

早期 T800 仅有两段前进行走，却继承 G1 的前后、横移和转向命令；这个问题已经修复。但侧步任务仍只有5个短片段，且没有通过闭环固定速度门禁，数据量与动态多样性可能仍不足。

### 6.4 当前不能确认最后一次速度奖励是否成功

最后一轮训练日志是健康的，但 `Metrics/twist/error_vel_xy≈0.83` 仍偏大；尚无固定左右命令的量化回放。因此目前最严谨的表述是“训练完成、行为未验收”，而不是“已经修复”。

### 6.5 深度相机不是当前第一阻塞

现阶段使用真实球状态都没有学出可靠侧移，直接加入16×9头部深度会同时引入感知、相机坐标和策略学习变量，无法定位失败层。应先通过 state-oracle。

## 7. 重要配置和源码地址

### 7.1 本地可直接查看

```text
总体设计：<LOCAL_WORKSPACE>\docs\superpowers\specs\2026-08-24-t800-pacman-state-oracle-design.md
实施计划：<LOCAL_WORKSPACE>\docs\superpowers\plans\2026-08-24-t800-pacman-state-oracle.md
侧步/避障设计：<LOCAL_WORKSPACE>\docs\superpowers\specs\2026-08-25-t800-task-c-forward-lateral-dodge-design.md
侧步/避障计划：<LOCAL_WORKSPACE>\docs\superpowers\plans\2026-08-25-t800-forward-lateral-dodge.md
运行手册：<LOCAL_WORKSPACE>\task_c_pacman_work\docs\t800_migration\runbook.md
质量门禁：<LOCAL_WORKSPACE>\task_c_pacman_work\docs\t800_migration\quality_gate.md
```

关键代码：

```text
T800机器人常量：task_c_pacman_work\src\assets\robots\engineai_t800\t800_constants.py
T800基础环境：task_c_pacman_work\src\tasks\amp_loco\config\t800\env_cfgs.py
T800避障环境：task_c_pacman_work\src\tasks\amp_loco\config\t800\dodge_env_cfgs.py
T800避障PPO/AMP配置：task_c_pacman_work\src\tasks\amp_loco\config\t800\dodge_rl_cfg.py
T800基础RL配置：task_c_pacman_work\src\tasks\amp_loco\config\t800\rl_cfg.py
CBF动作层：task_c_pacman_work\src\tasks\amp_loco\mdp\actions_cbf.py
Link-CBF：task_c_pacman_work\src\tasks\amp_loco\mdp\cbf.py
关节CBF：task_c_pacman_work\src\tasks\amp_loco\mdp\cbf_joint.py
目标命令：task_c_pacman_work\src\tasks\amp_loco\mdp\goal_command.py
奖励：task_c_pacman_work\src\tasks\amp_loco\mdp\rewards.py
终止条件：task_c_pacman_work\src\tasks\amp_loco\mdp\terminations.py
部署策略切换：task_c_pacman_work\worktree_sync\deploy\t800_policy_switch.py
```

### 7.2 服务器数据和证据

```text
避障项目根目录：<SERVER_USER_DIR>/task_c_t800_dodge
动作CSV质量报告：<SERVER_USER_DIR>/task_c_t800_dodge/metrics/t800_csv_qa.json
动作来源清单：<SERVER_USER_DIR>/task_c_t800_dodge/SOURCE_MANIFEST.json
第一轮指标：<SERVER_USER_DIR>/task_c_t800_dodge/metrics/probe256_iter1000_metrics.json
最后侧步训练日志：<SERVER_USER_DIR>/task_c_t800_dodge/logs/lateral_vtrack_w3_std035_nomask4096_iter500_from45999.log
```

本地视频总目录：

```text
<LOCAL_WORKSPACE>\task_c_videos
```

其中建议重点看：

```text
probe999_gate\obstacle.mp4
resume2000_gate\obstacle_model2000.mp4
safevel2499_gate\obstacle_safevel2499.mp4
hardfilter2499_gate\obstacle_hardfilter2499.mp4
g1_original_walk_policy_vx0p5_reference.mp4
```

## 8. 推荐的下一步

1. 找到最后一轮 lateral-vtrack 的最终 checkpoint，做完全确定性的固定 `vy=+0.5` 和 `vy=-0.5` 各1000步评估。
2. 必须报告机体坐标平均/中位横向速度、前向漂移、航向变化、根高度、终止原因，并生成视频。
3. 若两侧都能稳定达到至少约0.3 m/s，再加入慢球课程，然后逐步恢复正常球速。
4. 若仍然站立不动，不再继续从原 dodge checkpoint 微调；改为显式的左/右技能条件、更多侧步片段，或分离左右两个低层技能。
5. state-oracle 有球门禁通过后，才加入头部深度和感知网络。
6. 每个阶段都应保留同起点 A/B，避免同时修改动作数据、奖励、CBF开关和PPO。

## 9. 待进一步分析的问题

1. 为什么 `track_anchor_linear_velocity` 提高权重后仍可能出现高 episode length、低真实横向速度？
2. 现有 reward 中是否存在站立/AMP/存活项远大于速度误差项，导致局部最优？
3. 应该让 CBF 过滤后的速度进入 actor observation，还是只作为 shield/动作后处理？
4. 5段短侧步动作对 AMP 是否足够，是否需要相位、方向标签或左右分离训练？
5. 采用双策略切换时，怎样避免 locomotion 与 dodge 策略的观测历史和动作历史不连续？
6. 在不引入深度相机的前提下，最小可证伪的 state-oracle 课程应如何设计？

## 10. 不应做出的错误结论

- 不能说“T800 CBF 已经成功避障”；目前没有通过有球侧移门禁。
- 不能说“CBF 算法无效”；当前首先缺少可执行的侧步技能和明确学习信号。
- 不能用训练 reward 或 episode length 代替固定命令下的真实速度与视频。
- 不能把早期前倒全部归因于 PPO；已确认存在根俯仰异常、脚底悬空和命令分布错配。
- 不能直接进入相机阶段，否则会同时混入感知和控制问题。
