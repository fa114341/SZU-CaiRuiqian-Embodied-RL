<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 23. Do not use as a complete reproduction guide. -->

# RoboStriker T800 项目：进度、卡点与技术交接

> 更新日期：2026-09-16
> 用途：供后续读者分析 RoboStriker 的三阶段训练、T800 当前 Stage2/Stage3 卡点和下一步实验。
> 安全说明：本文不含服务器 IP、密码、Token 或私钥。

## 1. 一句话结论

官方 G1 Stage3 双机器人格斗推理已经成功复现；T800 的 Stage1 九动作跟踪也已通过，Stage2 九动作蒸馏完成1000轮且总体曲线很好。但逐技能 Oracle 证明 Stage2 的后退和转向 latent 仍不能稳定驱动 decoder，因此当前 Stage3 latent-teacher 门禁失败，Stage3 正式训练没有启动。现阶段真正的阻塞在 Stage2 技能可执行性，而不是继续调 Stage3 出拳奖励。

截至本次整理，服务器 GPU 空闲，没有 RoboStriker 训练进程或 tmux 会话在运行。

## 2. RoboStriker 三个 Stage 的准确含义

### Stage1：动作跟踪 Teacher

输入机器人状态和未来参考动作，PPO学习直接输出25维 T800 关节动作。它回答的是：“给定某段专家动作，机器人能否稳定跟踪？”

### Stage2：Encoder / Prior / Decoder 蒸馏

冻结 Stage1 teacher，用动作编码器把专家状态压成32维 latent；decoder根据本体状态和latent输出25维动作；causal prior只看机器人自己的历史，学习预测latent。它不是新的任务强化学习，而是把动作跟踪能力压缩成可复用的技能空间。

### Stage3：双机器人任务强化学习

高层 task policy根据对手、距离、朝向、接触等信息输出32维task latent；Stage2 prior提供32维先验latent，两者相加后送入冻结的Stage2 decoder，最后输出25维关节动作。Stage3用PPO、自博弈或沙袋课程学习“何时选择哪种技能”。

关键依赖关系：

```text
Stage1动作跟踪正确
→ Stage2 encoder/prior/decoder能稳定执行各技能
→ Stage3才能通过latent选择接近、转向、后退和出拳
```

如果 Stage2 的后退/转向 latent 本身不稳定，Stage3 再好的距离/朝向奖励也无法选出可执行技能。

## 3. 仓库、环境和当前分支

```text
官方仓库：<SERVER_USER_DIR>/RoboStriker-Preview
官方基线commit：52f54137770ff632391dd091e973ceb098ba3f98
T800 worktree：<SERVER_USER_DIR>/.config/superpowers/worktrees/RoboStriker-Preview/t800-stage1-tracking
分支：repro/t800-stage1-tracking
Python/Isaac环境：<SERVER_USER_DIR>/isaac_env
Isaac Lab：<SERVER_USER_DIR>/IsaacLab-c227752
T800 USD：<SERVER_USER_DIR>/engineai_rl_lab-main/source/engineai_rl_lab/engineai_rl_lab/assets/t800/serial_t800.usd
RTX 5090 GL补充库：<SERVER_USER_DIR>/isaac_runtime_libs/usr/lib/x86_64-linux-gnu
```

当前已提交到：

```text
1bc0641 feat: register Stage3 latent teacher curriculum
```

之后的 Stage2 latent 精确导出修复、Oracle launcher和测试仍有未提交修改。worktree还包含大量早期实验文件和其他脏改动，所以任何提交必须只 stage 精确文件，不能全量 `git add .`。

## 4. 官方 G1 复现结果

官方 `model_3000.pt` 的 G1-vs-G1 Stage3 推理已跑通。只做了两处兼容修复：任务注册导入，以及 Tensor `.copy()` 改成 `.clone()`。900帧/30秒视频中，两台G1会接近、缠斗、出拳/踢击、击倒并复位，不是固定动作回放。

```text
官方任务：Combat-Stage3-Selfplay
checkpoint：logs/rsl_marl/g1_flat_rsl_rl_ppo_stage3/combat/model_3000.pt
服务器视频：<SERVER_USER_DIR>/robostriker_stage3_repro_20260902/robostriker_g1_stage3_official_checkpoint_seed42_900f.mp4
本地视频：<LOCAL_WORKSPACE>\robostriker_g1_stage3_official_checkpoint_seed42_900f.mp4
```

这证明官方 Stage3 checkpoint 和双机器人环境有效，但预览仓库缺部分 boxing motion/BoxingRing 资产，因此不能称为完整复现官方 Stage1→Stage2→Stage3 全训练链。

## 5. T800 接入和 Stage1 进度

### 5.1 双T800和低层合同

双T800 Isaac Lab环境、25维动作、接触传感器、手—躯干接触和视频链路已跑通。早期还把旧 PAC-MAN locomotion `model_45000.pt` sim2sim 到 Isaac Lab，验证两台T800可以稳定前进；这只用于证明动力学/观测接入，不是格斗策略。

### 5.2 动作数据路线

自录动作采用：

```text
人体视频
→ GVHMR恢复人体动作
→ 官方GMR直接 human→T800
→ 时间平滑、速度/限位门禁
→ 50 Hz RoboStriker NPZ
```

没有走 human→G1→T800 的二次关节映射。

九动作集：

```text
backward
forward
left_right_combo
left_straight
right_straight
side_step_left
side_step_right
turn_left
turn_right
```

### 5.3 九动作 Stage1 最终结果

比较了旧七动作checkpoint续训九动作和九动作从头训练。两者分别对九个动作评估501步，均为0 reset、0 termination、0 truncation；从头训练版综合误差略低，因此选它作为Stage2唯一Teacher。

```text
动作目录：<SERVER_USER_DIR>/robostriker_t800_turn_motion_20260914/training/npz_stage1_turn9_v1
manifest：<SERVER_USER_DIR>/robostriker_t800_turn_motion_20260914/training/stage1_turn9_v1_manifest.json
[REDACTED: credential or access instruction]
Teacher：上述目录/model_4223.pt
[REDACTED: credential or access instruction]
评估：<SERVER_USER_DIR>/robostriker_t800_turn_motion_20260914/eval_stage1_scratch4223_v1
[REDACTED: credential or access instruction]
```

Stage1 结论：九个动作的直接跟踪能力已通过当前门禁，尤其原始转向素材本身不是错标签。

## 6. Stage2 九动作结果

Stage2配置：

```text
任务：RoboStriker-Tracking-Flat-T800-v0-stage2-latent
环境数：4096
轮数：1000
seed：42
future frames：4
保存间隔：250
上肢action scale：肩/肘pitch-roll-yaw=0.8，肘yaw=0.26
```

运行与最终模型：

```text
[REDACTED: credential or access instruction]
checkpoint：上述目录/model_999.pt
[REDACTED: credential or access instruction]
```

最终聚合指标（末50轮附近）：

```text
Mean episode length：约487.75 / 500
Episode_Termination/ee_body_pos：约0.0476
Mean reward：约42.24
behavior loss：约0.0240
KL：约0.417
latent_reg loss：约0.00437
```

这些值看起来健康，但adaptive sampler中 backward（traj0）长期为最难动作，末期top1概率约0.196。聚合平均掩盖了具体技能失败。

## 7. 重要维度合同

当前九动作 Stage2 checkpoint：

```text
Stage1 teacher observation：376
teacher action：25
student proprio：78
latent：32
student/decoder策略输入概念上：78 + 32
encoder history：2 × 511 = 1022
prior history：10 × 78 = 780
```

当前 Stage3 latent-teacher 环境实际观测：

```text
task policy：94
critic：393
prior：[10, 78]
decoder proprio：78
actions：25
```

关节和body顺序在运行时由显式validator检查；不要凭字符串排序手工重排。PhysX枚举顺序不等于简单的`J00,J01,...`文本排序。

## 8. 以前 Stage3 做过什么，以及为什么仍不够

### 8.1 旧版本主要问题

早期Stage3会高速贴身、冲撞、摔倒，原因包括latent噪声过大、单向速度奖励、躯干碰撞可刷奖励和缺少稳定约束。后续逐步加入：

- latent噪声限制；
- AMP warm-up；
- balance、自身倒地惩罚；
- torso facing；
- 距离带和相对闭合速度；
- 固定prior物理沙包；
- 非对称特权观测；
- 去掉“对手倒地给正奖励”的shortcut。

### 8.2 10小时非对称感知正式训练

4096环境×2000轮，共196,608,000环境步，训练状态0，最终episode length约1927。但500步评估始终在约1.07–1.48 m，0帧进入0.70–0.95 m拳距，`hit_reward`和`complete_punch_reward`均为0。

结论：稳定、朝向和不相撞改善了，但策略落在“远处护架不出拳”的安全局部最优。

### 8.3 固定沙包和距离奖励实验

固定prior对手能显著提升拳距占比，但仍出现贴身推挤、推倒或后半程分离；调整距离高斯只是在“近距离挤压”和“远距离无梯度”之间移动问题。多轮短探针都没有产生真实有效出拳。

因此最新路线不再继续堆奖励，而是让Stage3先用已知Stage2技能latent学习“过近后退、偏航转向”，再逐步撤教师并开启出拳。

## 9. 当前 latent-teacher 实现

已经实现并提交的模块：

- Stage3 per-env teacher状态机。
- Stage2 latent bank加载和SHA校验。
- 固定物理沙包、无出拳奖励的环境。
- rollout存储teacher latent/mask。
- 带mask的latent回归损失。
- runner集成、调度与TensorBoard指标。
- 新任务注册：`Combat-T800-Stage3-LatentTeacher-FixedSandbag-v0`。

已提交commit：

```text
03c983a feat: add Stage3 latent teacher controller
c05e596 feat: export Stage2 teacher latent bank
3346488 feat: add punch-free latent teacher environment
83985d0 feat: store Stage3 latent teacher transitions
8b8ea31 feat: train Stage3 with masked latent teacher PPO
cf11818 feat: integrate latent teacher into Stage3 runner
1bc0641 feat: register Stage3 latent teacher curriculum
```

latent状态机的目标逻辑：

```text
距离过近 → backward latent
对手在左/右偏航 → turn_left / turn_right latent
位置与朝向合格 → guard latent（零teacher target）
```

bank成功导出：

```text
根目录：<SERVER_USER_DIR>/robostriker_t800_turn_motion_20260914/stage3/latent_teacher_v1_20260916/bank_v1
backward：[212,32]
turn_left：[132,32]
turn_right：[207,32]
[REDACTED: credential or access instruction]
[REDACTED: credential or access instruction]
```

导出过程有一个重要实现细节：连续closed-loop回放会因当前Stage2技能不稳而提前终止，不能用来导出完整专家序列；当前导出器对每个源帧做精确teacher-forced reset，再编码该帧latent，并在单一SimulationContext中处理所有动作。

## 10. Oracle 门禁结果

Oracle不是训练出的Stage3 policy，而是直接让状态机选择Stage2 latent并喂给冻结decoder。若Oracle都不能完成后退/转向，Stage3不可能靠学习选择同一批latent解决问题。

输出目录：

```text
<SERVER_USER_DIR>/robostriker_t800_turn_motion_20260914/stage3/latent_teacher_v1_20260916/oracle_gate_v1
```

结果：

| 场景 | 期望 | 实际 | 结论 |
| --- | --- | --- | --- |
| too_close | 从0.55 m后退到安全距离 | 一度到0.717 m，但250步末回到约0.471 m；无reset | 后退不能持续保持 |
| yaw_left | 使用turn_right把约30°误差降下来 | 绝对yaw约0.524→0.650 rad，且1次falldone reset | 失败 |
| yaw_right | 使用turn_left把约30°误差降下来 | 绝对yaw约0.524→0.419 rad，但1次falldone reset | 有短暂改善但门禁失败 |

三段都没有非有限动作。Oracle gate正确写为 `passed=false`，所以Stage3 smoke/正式训练没有启动。

### 10.1 已排除的简单解释

- 原始动作标签没有颠倒：backward确实沿身体后向移动约1.96 m；turn_left约+85.7°；turn_right约-76.6°。
- controller转向符号正确；场景名和signed yaw需要注意，但所选动作方向符合定义。
- bank shape、float、50 Hz、归一化和checkpoint SHA均通过。

### 10.2 进一步证据

直接用官方Stage2 encoder路径回放 backward 20步，发生4次 `ee_body_pos` reset；从frame 40开始仍是4次。说明问题不是Stage3状态机，而是当前Stage2 encoder latent/decoder不能稳定闭环执行后退。

这与训练中 backward 长期被adaptive sampler判为最难动作一致。

## 11. 当前最可信的根因

1. Stage1可以在完整参考和未来帧条件下跟踪动作，但Stage2把动作压到32维后，特定移动技能没有被decoder稳健复现。
2. Stage2整体episode length接近满分，掩盖了少数难动作；当前验收方式过度依赖总体均值。
3. backward和turning在动态闭环中可能需要更长训练、动作均衡/过采样、不同latent正则或更强behavior distillation。
4. Stage3之前的距离/朝向/出拳奖励确实有问题，但当前latent-teacher Oracle红灯证明，上游技能是更早的硬阻塞。

## 12. 重要源码和配置文件

### 12.1 本地镜像

本地镜像根：

```text
<LOCAL_WORKSPACE>\_remote_work\RoboStriker-Preview
```

当前latent-teacher设计与计划：

```text
docs\superpowers\specs\2026-09-16-t800-stage3-latent-teacher-curriculum-design.md
docs\superpowers\plans\2026-09-16-t800-stage3-latent-teacher-curriculum.md
```

关键代码：

```text
Stage2 latent导出：scripts\rsl_rl\export_t800_stage2_teacher_latents.py
Stage3探针/Oracle：scripts\rsl_rl\probe_t800_stage3.py
latent teacher状态机：source\whole_body_tracking\whole_body_tracking\rsl_rl\rsl_rl\utils\latent_teacher.py
Stage3 runner：source\whole_body_tracking\whole_body_tracking\rsl_rl\rsl_rl\runners\on_policy_runner_stage3.py
Stage3 PPO：source\whole_body_tracking\whole_body_tracking\rsl_rl\rsl_rl\algorithms\ppo_stage3.py
Stage3 storage：source\whole_body_tracking\whole_body_tracking\rsl_rl\rsl_rl\storage\rollout_storage_stage3.py
Stage3 actor/critic：source\whole_body_tracking\whole_body_tracking\rsl_rl\rsl_rl\modules\actor_critic_stage3.py
T800 Stage3环境：source\whole_body_tracking\whole_body_tracking\tasks\combat\config\t800_vs_t800\t800_stage3_env_cfg.py
T800 Stage3 PPO配置：source\whole_body_tracking\whole_body_tracking\tasks\combat\config\t800_vs_t800\agents\rsl_rl_ppo_cfg.py
T800任务注册：source\whole_body_tracking\whole_body_tracking\tasks\combat\config\t800_vs_t800\__init__.py
奖励实现：source\whole_body_tracking\whole_body_tracking\tasks\combat\mdp\rewards.py
回归测试：tests\test_t800_stage3_latent_teacher.py
Stage3历史测试：tests\test_t800_stage3_sandbag.py
```

启动器：

```text
九动作Stage2：launch_t800_turn9_stage2_scratch4223_4096_iter1000_20260915.remote.sh
latent bank导出：launch_t800_stage3_latent_teacher_export_20260916.remote.sh
Oracle门禁：launch_t800_stage3_latent_teacher_oracle_eval_20260916.remote.sh
```

### 12.2 当前测试状态

latent导出精确帧、单SimulationContext和Oracle launcher测试加入后，完整快速测试为 `322 passed`。但Task8相关的导出器、测试和Oracle launcher尚未提交；运行前应检查diff与语法。

## 13. 推荐的下一步

1. 暂不启动Stage3，因为Oracle红灯已经证明上游技能不可用。
2. 从当前九动作Stage2 `model_999.pt` 做一次独立续训对照，建议再1000轮、4096环境、配置完全不变；上次1000轮实测约1小时40分。
3. 续训前后都必须对 backward、turn_left、turn_right 做固定起点的encoder和prior逐技能评估，不能只看总体mean。
4. 若续训后Oracle通过，再重导bank并依次跑三个Oracle场景；只有0 reset、距离/偏航朝正确方向收敛才启动Stage3。
5. 若续训仍失败，停止堆轮数，改做难动作过采样、动作单独Stage2、behavior loss/latent正则A/B，或检查future-frame teacher到causal decoder的信息瓶颈。
6. Stage3首次通过时只做无出拳的后退/转向选择课程；之后逐步撤teacher，最后才开启出拳奖励和动态对手。

## 14. 待进一步分析的问题

1. 为什么Stage2总体episode length接近500，但backward encoder回放20步会reset 4次？
2. 当前Stage2 loss是否按motion均衡，还是长/容易动作主导batch平均？
3. 对难动作应该优先续训、过采样、单动作蒸馏，还是改变latent/decoder容量？
4. `future4` Stage1 teacher到只看proprio+latent的Stage2 decoder之间是否存在不可补偿的信息丢失？
5. teacher-forced逐帧latent bank能否代表闭环技能，还是应导出一个稳定的skill prototype/phase-conditioned controller？
6. Stage3 teacher应该监督task latent本身，还是监督task latent + prior latent之和？
7. 如何制定逐技能门禁，避免再次被总体训练指标掩盖？

## 15. 不应做出的错误结论

- 不能说“T800 Stage3已经开始正式训练”；当前被Oracle门禁阻止。
- 不能说“只是Stage3奖励没调好”；Stage2后退/转向闭环已经独立失败。
- 不能说“Stage2完全失败”；整体蒸馏、站立和大部分动作正常，问题集中在关键难技能。
- 不能把Stage1跟踪成功直接等价为Stage2 latent技能成功。
- 不能因为某个转向场景短暂到达小yaw就放行；发生reset后的最优值不算有效成功。
- 不能直接从失败的Stage3 checkpoint续训来验证新课程；应从已验证Stage2 checkpoint新建Stage3。
