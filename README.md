# 深圳大学-蔡锐潜-具身强化学习/众擎机器人格斗大赛开源

这里整理了我在 T800 上做过的几条实验线：人体动作恢复与重定向、行走和动作跟踪、避障侧移，以及 UFO、RoboStriker 的格斗动作与分层策略。仓库保留了有用的配置、实验记录和仿真视频，也写明哪些尝试没有跑通。它不是把几个上游项目拼成的统一训练框架。

## 实验记录

| 方向 | 记录与结果 |
| --- | --- |
| [人体视频 → GVHMR → GMR → T800](projects/motion-pipeline.md) | 做过自录素材的动作恢复与重定向；需要逐段检查关节方向、接地和时间连续性。 |
| [SMP / T800 行走](projects/locomotion.md) | 留有 12 段行走的历史交付记录；后续 10 段版本的权重不在仓库内。 |
| [EngineAI RL Lab 动作跟踪](projects/action-tracking.md) | 庆祝、挑衅和自录动作实验；动作质量受素材与重定向影响明显。 |
| [PAC-MAN / CBF 避障与侧移](projects/cbf-pacman.md) | 侧移技能有阶段性结果，完整避障仍有直走、转向与稳定性问题。 |
| [UFO / T800](projects/ufo.md) | 记录了 50M/55M 训练与评估；前进动作出现后退、小碎步等语义偏差。 |
| [RoboStriker / T800](projects/robostriker.md) | 跑过 Stage1/2/3；Stage3 的对准、拳距和有效出拳还没有达到预期。 |

## 从哪里开始看

- [`projects/`](projects/) 是每条实验线的简要记录；参考仓库集中列在 [`UPSTREAM.md`](UPSTREAM.md)。
- [`media/`](media/) 放了筛选过的机器人仿真视频。视频对应具体实验阶段，不代表已经得到可部署策略。
- [`docs/archive/`](docs/archive/) 是历史工作文档的公开整理版，其中的路径和部分内容经过脱敏，不能直接当作一键复现说明。
- [`MODELS.md`](MODELS.md) 说明哪些权重曾经训练出来、哪些文件实际收录；[`SOURCE_BACKLOG.md`](SOURCE_BACKLOG.md) 记录尚缺的源码和配置。

旧版行走配置的 25 维 action 顺序和末端顺序单独放在 [`configs/historical_locomotion_joint_order.json`](configs/historical_locomotion_joint_order.json)。例如，在复用旧策略前可以先检查维数和腰部 link 名称：

```python
import json
from pathlib import Path

cfg = json.loads(Path("configs/historical_locomotion_joint_order.json").read_text(encoding="utf-8"))
assert len(cfg["action_and_joint_order"]) == 25
assert cfg["link_mapping"]["waist_link"] == "LINK_WAIST_YAW"
```

这份配置只对应历史 12 段版本，不能直接套到后续 10 段权重上；其中左肩 roll 的历史命名也有误，详见配置文件注释。

## 发布范围

目前公开的是本机留存的文档、配置和少量仿真视频。服务器上的最新权重、完整训练源码与日志没有归档，不能据此声称所有结果都可完整复现。原始人像视频、受限模型和数据集、连接信息及密钥均未收录。文件校验值见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。

上游源码和数据仍归各自作者所有。本仓库尚未附统一开源许可证；公开可读不等于所有文件均可自由再分发，具体边界见 [`RIGHTS.md`](RIGHTS.md) 和 [`PUBLICATION_POLICY.md`](PUBLICATION_POLICY.md)。这些实验也未经实机部署验证。
