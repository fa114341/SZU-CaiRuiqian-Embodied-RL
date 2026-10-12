# 任务 B：T800 动作跟踪

在 EngineAI RL Lab 的 T800 任务上训练庆祝、挑衅和自录动作。动作用 BVH 或人体视频恢复得到，再重定向到 T800，检查关节限位、起止姿态、脚底接地、碰撞与动作语义，然后进入动作跟踪 PPO。

历史实验包含单动作探针、4096 环境正式训练、若干自录视频的碰撞保护和评估。不同动作质量差异较大，尤其视频恢复与重定向误差不能依靠延长 PPO 训练自动消除。公开仓库不放原始出镜视频、受限人体模型或未经许可的第三方动作数据。

上游：[EngineAI RL Lab](https://github.com/engineai-robotics/engineai_rl_lab)、[GVHMR](https://github.com/zju3dv/GVHMR)、[GMR](https://github.com/YanjieZe/GMR)。
