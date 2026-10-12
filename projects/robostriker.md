# RoboStriker / T800 分层格斗

研究分为：Stage1 动作跟踪技能（直拳、组合拳、前后左右移动、转身等），Stage2 Encoder/Prior 压缩与条件切换，Stage3 双机器人交互策略。上层需要决定朝向、距离和何时选技能，低层必须能稳定执行转身、后退和出拳。

历史视频暴露出 Stage1 抖动、拳不够直，Stage2 转身→后退切换不稳定，Stage3 初始正面对准的指标可能掩盖真实转向能力，且多次未进入有效拳距或发生碰撞/摔倒。Stage3 尚未达到“自主对准并有效出拳”的质量门槛，更不能称为可直接实机部署。

本项目的机器人仿真精选视频保留了部分成功与失败的过程，便于对照。Stage1/2/3 权重主要仍在服务器，见 [模型登记](../MODELS.md)。

上游：[RoboStriker-Preview](https://github.com/yinkangning0124/RoboStriker-Preview)。
