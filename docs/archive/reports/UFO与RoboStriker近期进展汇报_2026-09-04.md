<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 8. Do not use as a complete reproduction guide. -->

# UFO 与 RoboStriker 近期进展汇报

汇报范围：2026-08-30 至 2026-09-04
对象：URKL 小干群 / T800 自主格斗方向
官方材料：[UFO 论文](https://roboparty.github.io/UFO/assets/UFO.pdf)｜[UFO 项目](https://roboparty.github.io/UFO/)｜[RoboStriker 项目](https://yinkangning0124.github.io/RoboStriker/)｜[RoboStriker 论文](https://arxiv.org/abs/2601.22517)

> 动态状态：截至 2026-09-04 下午，T800 RoboStriker Stage2 的4096环境×1000轮训练及4轨迹encoder/prior双模式门禁均已完成。encoder初始化后额外reset由旧探针的5次降至0次；prior由21次降至8次，改善约62%，但prior仍非完全稳定。

## 一、1 分钟 3P 汇报版

🤖 URKL 小干群—T800 格斗策略（8月30日—9月4日）
Progress: 已分别打通 UFO-FB 的 T800 配置、真实参数更新、checkpoint 与推理链路，以及 RoboStriker 官方双 G1 Stage3 推理、双 T800 环境、T800 Stage1 格斗动作跟踪和 Stage2 latent 蒸馏链路。RoboStriker T800 Stage1 与 Stage2 均使用4096环境×1000轮、约9830万环境步完成训练；Stage2 encoder在4条固定轨迹上已全部连续存活，prior额外reset由21次降至8次。
Plans: 保存并展示4组encoder/prior并排视频；在明确prior剩余reset可接受范围后，进入T800 Stage3双机器人高层策略/自博弈。UFO后续继续检查初始相位、latent和单动作数据覆盖。
Problems: UFO 的 T800 和同预算 G1 小模型都会在约0.5–1秒倒地，增加到640次更新仍未改善，说明目前只能称工程链路跑通；RoboStriker Stage2 的encoder门禁已通过，但causal prior在1196个非初始化评估步中仍有8次 `ee_body_pos` 终止，不能表述为完全稳定。

## 二、开场可以这样讲

学姐发的两个链接实际上对应两条互补路线。UFO 关注的是如何用无监督强化学习预训练一个覆盖多种行为、可复用的低层行为基础模型；RoboStriker 关注的是如何把人体格斗技能、低层动作执行和双机器人高层博弈分层组合起来。对 URKL 来说，UFO 更像通用技能底座，RoboStriker 更像自主格斗的任务架构。

我近期不是只看了论文，而是分别做了可执行验证。UFO 这边已经把 T800 机器人配置、动作数据、训练更新、checkpoint 和推理视频串通，但当前小预算策略仍会倒地，所以没有宣称复现了论文效果。RoboStriker 这边先跑通官方双 G1 格斗，再迁移 T800：已有稳定的双机行走环境、4段 T800 格斗动作的 Stage1 跟踪策略，并完成 Stage2 正式训练与双模式门禁；encoder四段稳定，causal prior显著改善但仍有少量终止，下一步才是双机器人 Stage3 自博弈。

## 三、两个项目分别解决什么

| 项目 | 核心问题 | 方法主线 | 对 URKL 的价值 |
| --- | --- | --- | --- |
| UFO | 不为每个动作单独设计奖励，如何获得通用、鲁棒、可组合的行为能力 | UFO-FB、TeCH、off-policy 探索、多动作数据、latent 条件策略 | 建立行走、起身、抗扰、全身动作等通用低层技能底座 |
| RoboStriker | 高维接触条件下，如何同时解决人形动作质量和双机器人竞争决策 | Stage1 动作跟踪 → Stage2 latent 蒸馏 → Stage3 LS-NFSP 自博弈 | 直接提供“低层格斗技能 + 高层对手决策 + 自博弈”的比赛路线 |

共同点是都不希望高层直接在几十维关节动作上盲目探索，而是先得到受人体运动约束的动作空间。区别是 UFO 追求通用行为表示，RoboStriker 明确面向拳击竞争任务。

## 四、UFO：论文理解与实际复现

### 4.1 论文核心

UFO 是一个人形机器人无监督强化学习框架，不只是一个算法：

1. UFO-FB 对 Forward-Backward 行为表示做了高吞吐、分布式、off-policy 工程优化；官方页面称高端多卡训练可缩短到约6–8小时。
2. TeCH 用对比学习建立反映“时间可达性”的潜空间，并用状态朝潜目标前进的距离变化提供稠密奖励。
3. 框架抽象了机器人形态和 RobotState 动作接口，支持多机器人和多数据源；但“通用”不代表 G1 checkpoint 能直接套到 T800，目标机器人仍需专属配置、专属重定向数据和重新训练。
4. 论文展示了动作跟踪、目标到达、跌倒恢复、技能注入和实机抗扰遥操作。

“无监督”不是无数据、无奖励，而是不需要为走路、下蹲、恢复等每个下游任务分别提供人工标签和任务专属奖励；训练仍使用无标签动作数据、内在表示奖励和安全约束。

### 4.2 我完成了什么

- 固定并验证官方 UFO 代码与 G1 LaFAN 数据。
- 为 RTX 5090 修复完整 CUDA 12.8 依赖组合，解决真实更新时的 `sm_120/NVRTC` 不兼容。
- 接入 T800 的 MJCF/URDF、25关节动作顺序、root/feet/hands/torso 语义、执行器和限位。
- 修复 UFO MotionLib 不能读取 T800 拆分式 MJCF `<include>` 的问题，同时保留 G1 回归；完整快速测试97项通过。
- 使用人体视频经 GMR **直接重定向到 T800** 的 RobotState 数据，完成 CPU 环境、GPU 真实 optimizer update、checkpoint 保存/加载和300帧推理视频。
- 做了同预算 G1 对照：G1 与 T800 在12288环境步、32次更新下都会很快倒地，因此不能把该现象直接归因于 T800 关节映射。
- 把 T800 增加到51200环境步、640次更新，效果仍未形成有意义改善。

### 4.3 当前结论

准确表述是：**T800 的 UFO-FB 工程闭环已经跑通，但论文级行为质量尚未复现。**

当前 T800 只用了1段约10秒动作和极小训练预算；32次与640次更新的视频都约0.17秒开始偏离、约0.5–1秒倒地。这个结果否定了“只从32增加到640次更新就能解决”的假设，但还不能断言 UFO 算法无效，也不能证明只是训练量问题。下一步需逐项检查 reset 目标相位、tracking latent、动作数据覆盖、初态和动力学配置。

## 五、RoboStriker：项目理解与实际复现

### 5.1 三阶段架构

1. **Stage1：单机器人动作跟踪。** 用人体动捕训练低层策略，获得拳击、闪避、步法等动作技能。
2. **Stage2：latent 蒸馏。** 冻结 Stage1 teacher，把动作能力压缩进结构化 latent；prior 只看本体感知历史，decoder 将 latent 转成关节目标。
3. **Stage3：LS-NFSP 双机器人自博弈。** 高层策略观察双方状态，在 latent 空间选择战术，不直接搜索全部电机关节动作。

这就是为什么不能直接从“双机器人强化学习”开始：如果没有稳定低层动作先验，高层探索会同时承担站立、运动、接触和战术学习，训练非常不稳定。

### 5.2 官方 G1 复现

- 修复预览仓库的两处兼容问题：Gym任务未注册，以及 Tensor 错用 `.copy()`。
- 官方 `model_3000.pt` 成功完成双 G1 Stage3 推理。
- 录像900帧、30秒，包含接近、缠斗、出拳/踢击、击倒和下一回合，不是固定动作回放。
- 这证明的是官方 G1 checkpoint 推理链路，不代表 Stage1→Stage3 全训练链都已复现；预览仓库没有发布完整 Stage1/Stage2 训练权重和动作目录。

### 5.3 T800 迁移进展

**环境和行走：**

- 建立双 T800 Isaac Lab 格斗环境，两机器人各25维动作和独立接触传感器。
- 把已有 T800 locomotion ONNX 从 MuJoCo sim2sim 到 RoboStriker/Isaac Lab。
- 双机可稳定向前走满300步；但现有行走策略没有转向、横移和对手追踪，面对面时会擦身而过，因此只能作为低层前进技能。

**Stage1 格斗动作：**

- 没有采用 G1→T800 二次映射，而是走 `LaFAN1人体BVH → GMR engineai_t800 → T800 CSV/NPZ`。
- 从候选中筛选4段各8秒的护架、步法/闪避、轻攻击和抬膝动作，共1596个50 Hz帧。
- 4096环境×1000轮、98,304,000环境步训练完成。
- Mean reward由前50轮均值0.22升到后50轮20.08，Mean episode length由16.27升到473.78。
- 4段固定动作均只在第1步出现初始化reset，第2..300步连续存活；画面无明显前倾、悬空、小臂反向或锁死。
- 局限：上肢幅度偏小，且部分轨迹有明显全局 anchor 平移漂移，不能称为高精度动作复现。

**Stage2 latent 蒸馏：**

- 已实现 T800 原生 Stage2 任务和 encoder/prior 双模式评估器；Stage2 是蒸馏，不是 PPO。
- 256环境×500轮小探针中，encoder 4条轨迹共5次额外reset，prior共21次，质量未通过。
- 进一步复核证明 Stage1 teacher 四条轨迹都稳定，Stage2 checkpoint 内 teacher 权重与 Stage1 完全一致；问题位于 student 模仿与闭环鲁棒性。
- teacher 控制的影子探针中，prior 与 encoder latent 余弦均值0.993，两者动作MSE只有0.0031；encoder相对teacher动作MSE为0.0404，prior相对teacher为0.0477。说明 prior 顺序、GMR 或严重输入混叠不是当前第一嫌疑，主要问题是500轮训练量下共享 student 仍不够准确，小误差在闭环中被放大。
- 保持官方算法不变完成4096环境×1000轮样本量门禁：共98,304,000环境步，最终Mean reward 25.21、Mean episode length 451.72，训练状态0且日志无致命错误。
- 最终model999的4轨迹双模式评估中，encoder额外reset为0（旧探针5）；prior额外reset为8（旧探针21），改善约62%。分轨迹prior为 `1/3/2/2` 次，均由 `ee_body_pos` 触发、无truncation。

### 5.4 当前动态状态

最终训练与评估结果：

| 指标 | 第0轮 | 第999轮 |
| --- | ---: | ---: |
| Mean reward | 0.30 | 25.21 |
| Mean episode length | 12.68 | 451.72 |
| Behavior loss | 1.2943 | 0.0291 |
| KL loss | 0.9633 | 0.4719 |
| `ee_body_pos` termination | 1.0000 | 0.1998 |
| Total timesteps | 98,304 | 98,304,000 |

训练耗时约1小时31分，无 Traceback、CUDA OOM、RuntimeError 或 NaN。固定视频门禁中encoder四段均无额外reset，说明student闭环稳定性已通过；prior额外reset由21次降至8次，已达到“显著改善并可进入下一阶段验证”的原定门槛，但不能说prior已经完全稳定。

## 六、对 URKL 比赛的路线判断

我目前认为 RoboStriker 比 UFO 更直接对应自主格斗主线，但二者并不冲突：

```text
人体格斗动作
→ 直接重定向到 T800
→ Stage1 稳定动作跟踪
→ Stage2 得到可供高层调用的 latent/decoder
→ 双 T800 环境中的对手观测、接触与击倒奖励
→ Stage3 高层 latent 策略和自博弈
```

UFO 可作为未来更通用的低层技能基础，例如起身、抗扰和多方向运动；RoboStriker 先解决“如何把技能用于对手条件下的决策”。当前不应该把官方29维G1 checkpoint硬映射成25维T800，也不应该在没有通过Stage2稳定门禁时直接宣称已经可以训练实战格斗策略。

## 七、汇报时建议展示的材料

按以下顺序展示，控制在3–5分钟：

1. **官方 RoboStriker 双 G1 效果**：说明官方三阶段方法确实能产生互动格斗，而非动作回放。
   [robostriker_g1_stage3_official_checkpoint_seed42_900f.mp4](</<LOCAL_WORKSPACE>/robostriker_g1_stage3_official_checkpoint_seed42_900f.mp4>)
2. **T800 Stage1 动作**：说明人体→T800重定向、动作跟踪和机器人稳定性已经打通。
[REDACTED: credential or access instruction]
   [4段动作抽帧](</<LOCAL_WORKSPACE>/robostriker_t800_stage1_model999_4clips_contact.jpg>)
3. **T800 双机稳定前进**：说明低层 locomotion sim2sim 能在双机环境运行，但主动指出不能转弯追人。
   [robostriker_t800_sim2sim_final_height10145_300f.mp4](</<LOCAL_WORKSPACE>/robostriker_t800_sim2sim_final_height10145_300f.mp4>)
4. **Stage2 500轮失败对比**：说明发现问题后做了 encoder/prior 拆分诊断，而不是盲目长训。
   [robostriker_t800_stage2_probe500_dualmode4_contact.jpg](</<LOCAL_WORKSPACE>/robostriker_t800_stage2_probe500_dualmode4_contact.jpg>)
5. 如果需要解释 UFO 当前质量门禁，再展示32次或640次更新版本，不要把倒地视频当成最终效果。
   [UFO T800 32次更新](</<LOCAL_WORKSPACE>/ufo_t800_stage1_fb_probe_tracking_0.mp4>)
   [UFO T800 640次更新](</<LOCAL_WORKSPACE>/ufo_t800_fb_budget51200_updates640_tracking_0.mp4>)
   [UFO G1 同预算对照](</<LOCAL_WORKSPACE>/ufo_g1_stage1_fb_probe_tracking_0.mp4>)

## 八、学姐可能追问的问题

### 1. 两个项目最大的区别是什么？

UFO 目标是通用无监督行为基础模型；RoboStriker 目标是自主拳击，使用动作跟踪、latent 技能空间和高层自博弈的三阶段结构。前者解决通用技能覆盖，后者解决对抗任务中的技能选择。

### 2. 为什么 UFO 叫无监督，明明使用了动作数据？

它不是无数据，而是不为每个下游任务分别提供人工标签和任务专属奖励。动作数据提供行为先验，FB/TeCH 产生表示或内在奖励。

### 3. UFO 的 FB 和 TeCH 有什么区别？

FB 学习可用于统一奖励优化的 forward-backward 表示；TeCH 重点学习状态间的时间可达距离，适合目标到达、轨迹跟踪和恢复，但不具备 FB 完全相同的统一奖励优化接口。

### 4. 为什么 UFO 还不能说复现成功？

目前只验证了 T800 软件链路和真实参数更新，数据只有约10秒、训练预算远低于论文亿级规模，而且策略会很快倒地；没有复现论文的 EMD/Joint MAE、恢复能力、TeCH、分布式训练和实机结果。

### 5. RoboStriker 为什么必须分三阶段？

Stage1先学物理可行的格斗动作，Stage2把动作压缩成结构化latent，Stage3才在低维latent空间学习对手策略。这样高层不需要重新探索站立和平衡，也减少直接在25个关节上做多智能体探索的不稳定性。

### 6. 为什么不能直接使用官方 G1 checkpoint？

G1是29维动作，T800是25维；两者的关节顺序、默认姿态、执行器、观测历史和body语义都不同。换机器人必须重新配置、用T800专属动作数据并重新训练。

### 7. T800 动作是不是先转到 G1 再映射？

不是。当前4段 RoboStriker 格斗数据来自 LaFAN1 人体BVH，经官方GMR直接优化到 EngineAI T800，再转换成 T800 NPZ；没有 G1 中间机器人。

### 8. Stage2 的问题到底是不是 prior？

500轮时 prior 闭环确实比 encoder 更差，但 teacher控制下 prior 与 encoder 的动作MSE仅0.0031；共享student相对teacher的MSE约0.0404更大。因此目前更符合“student未充分收敛，小误差被闭环放大”，不是已证明的 prior 输入顺序错误。

### 9. 为什么现在又跑1000轮，不是盲目加训练量吗？

不是直接上论文40000轮。500轮×256环境只有307.2万样本，而官方成功 checkpoint 的 normalizer 约见过1亿样本；现在4096×1000恰好约9830万样本，是有依据的样本量对照，并保持算法不变。完成后仍用相同4轨迹双模式门禁验证。

### 10. 现在离双 T800 自主格斗还有什么？

先让 Stage2 encoder/prior稳定调用低层技能；然后把双机器人对手观测、接触/击倒奖励和高层latent policy接到T800，最后进行Stage3自博弈和固定对手评估。目前还没有进入T800 Stage3，不能把动作跟踪视频称为自主格斗结果。

## 九、最后一句总结

这几天的核心进展不是“已经把两篇工作完整复现”，而是把两条路线的能力边界和 T800 迁移难点实测清楚了：UFO 已完成 T800 工程闭环但策略质量未通过；RoboStriker 已从官方双 G1 推理推进到 T800 Stage1 稳定格斗动作和 Stage2 正式训练。下一决定点是 Stage2 model999 的 encoder/prior 双模式门禁，只有通过后才进入双 T800 高层自博弈。
