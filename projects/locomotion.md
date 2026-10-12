# 任务 A：SMP → T800 locomotion

从 SMP 行走基线开始，完成 G1 数据到 T800 的运动映射、T800 模型接入、prior 预训练和 PPO 行走实验。曾比较部分数据、12 段与去除最严重 2 段后的 10 段版本，分析小臂关节反向、脚底悬空、身体前倾、动作数据速度与训练命令速度不一致等问题。

历史 12 段实验加入较温和的躯干前倾惩罚并设较大的免罚区，用户观看视频后认为较早版本改善明显。最终 10 段的模型、prior 和归一化文件与旧 12 段交付包不同；本地快照只有旧包，不能据此声明最终 10 段结果已归档。重定向的前倾统计和 gait 指标应随完整模型一起复核。

上游：[SMP](https://github.com/SUZ-tsinghua/smp)；机器人资料：[EngineAI T800](https://github.com/engineai-robotics/engineai_robotics_description)。
