<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 2. Do not use as a complete reproduction guide. -->

# T800 Locomotion 可复现交付包设计

> **2026-08-25 范围调整：** 用户要求“不用评估，简单搞一下”。因此本次不重新运行策略、不新增 ContactSensor/速度/滑移/支撑期/终止 JSON-NPZ、不编写定制 0.75 m/s 遥测器，也不实际构建 Docker。以下相关章节保留为原始完整方案记录，但不属于本次轻量交付。实际交付只整理已有 checkpoint、prior、norm、12 段 CSV/NPZ、源码与 T800 资产、实际 env/agent 配置、TensorBoard、现有视频、依赖锁文件、简明训练说明、manifest 和逐文件 SHA256；缺失项目如实标注，不补造。

## 1. 目标与最终成果定义

交付一个可独立传给第三方的自包含压缩包。唯一最终策略固定为用户已确认效果良好的 12 段训练：

- 训练名：`t800_oldgmr_tilt_w05_free30_ppo30000`
- 任务：`Smp-Forward-T800-Walk12-TiltPenalty`
- 环境数：4096
- 轮数：30000
- 最终检查点：`model_29999.pt`
- 动作数据：`t800_engineai_official_walk12_fullsmooth31_guard85`
- prior：与上述 12 段动作数据对应的 `pretrained.pt`
- 前倾正则：权重 `-0.5`，30° 免罚区

后续 10 段实验、任务 b、任务 c、失败探针和其他用户文件不进入本交付包。

## 2. 交付形态

输出目录与压缩包：

```text
T800_LOCOMOTION_REPRODUCE_12SEG/
T800_LOCOMOTION_REPRODUCE_12SEG.zip
```

包必须自包含，不依赖原服务器目录。所有可执行脚本从自身位置计算 `BUNDLE_ROOT`，不得硬编码服务器路径、用户名、IP、密码或本机桌面路径。

## 3. 包结构

```text
T800_LOCOMOTION_REPRODUCE_12SEG/
├── README_REPRODUCE.md
├── SOURCE_MANIFEST.json
├── SHA256SUMS.txt
├── artifacts/
│   ├── checkpoint/model_29999.pt
│   ├── prior/pretrained.pt
│   ├── normalization/norm_stats.npz
│   └── motions/
│       ├── csv/*.csv
│       └── npz/*.npz
├── source/smp/
├── configs/
│   ├── env.sanitized.yaml
│   ├── agent.sanitized.yaml
│   ├── JOINT_ORDER.md
│   ├── joint_order.json
│   └── source_provenance.json
├── scripts/
│   ├── train_reproduce.sh
│   ├── evaluate_vx0p75.sh
│   └── verify_bundle.py
├── environment/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── pip_freeze.txt
│   ├── system_info.json
│   └── dependency_notes.md
├── training/
│   ├── tensorboard/events.out.tfevents.*
│   ├── train.log
│   ├── curves.csv
│   ├── curves.png
│   └── run_artifacts.json
├── evaluation/
│   ├── metrics_summary.json
│   ├── metrics_timeseries.npz
│   ├── contact_sensor_schema.json
│   └── evaluation.log
└── media/
    ├── t800_locomotion_model_29999.mp4
    └── t800_locomotion_model_29999_contact_sheet.png
```

## 4. 源码真实性与本地修改

原训练目录没有 Git 元数据，最终 RSL-RL 运行目录也没有自动保存源码快照。因此不能仅凭当前目录声称它与训练时逐字节完全一致。

交付时采用以下证据链：

1. 以训练运行目录内的 `env.yaml`、`agent.yaml`、TensorBoard 和检查点作为实际生效配置与结果证据。
2. 比较服务器当前 `<SERVER_USER_DIR>/smp`、本地保存的 `smp-master/smp-master` 和文件修改时间；对每个被训练入口导入的源码文件记录来源、SHA256、修改时间及选择理由。
3. 将选定的完整源码和全部 T800/MjLab 资产复制到 `source/smp`，排除 `.venv`、缓存、日志、其他任务数据和用户秘密。
4. 使用公开 SMP 上游作为基线生成 `LOCAL_CHANGES.patch`；若无法唯一确定上游提交，则在 `source_provenance.json` 中记录仓库 URL、比较基线和这一限制，不伪造精确提交。
5. `SOURCE_MANIFEST.json` 对每个最终文件记录包内路径、大小、SHA256、类别、来源类别和净化状态。

## 5. 模型、prior、归一化和动作数据

必须同时包含：

- 最终 `model_29999.pt`；
- 与 12 段数据对应的 `pretrained.pt`；
- 对应 `norm_stats.npz`；
- 训练实际使用的 12 个 CSV 和 12 个 NPZ；
- 最终运行的 `env.yaml` 和 `agent.yaml` 的语义等价净化版本。

打包程序验证 CSV/NPZ 数量、文件名集合、数组有限性、norm 键和维度，并核对 prior/checkpoint 可被代码加载。不得混入 10 段 prior 或其 norm。

## 6. 顺序和映射说明

`JOINT_ORDER.md` 与 `joint_order.json` 必须由实际运行配置和模型资产生成，不手工猜测，并明确：

- policy action 顺序；
- articulation joint 顺序；
- observation 中 joint position/velocity 顺序；
- prior 特征使用的关节或 body 顺序；
- end-effector/body 顺序；
- 每项的零基索引、名称和左右侧；
- action 到 joint 的一一对应关系和 action scale/default offset；
- `LINK_HEAD_YAW` 是头部 yaw 链接，不得作为腰部锚点；
- `LINK_WAIST_YAW` 是腰部/躯干 yaw 锚点，其在 prior、EE/body 特征或奖励中的实际用途；
- 若代码资产实际名称与上述别名不同，记录真实名称、别名和解析来源。

生成后必须与检查点输出维度、MJCF/URDF joint 数和运行时 entity 顺序交叉验证。

## 7. 一键训练与固定速度评估

`train_reproduce.sh`：

- 从包内相对路径读取 12 段数据、norm 和 prior；
- 默认复现最终任务、4096 环境、30000 轮和实际 PPO/奖励配置；
- 允许通过环境变量覆盖环境数和轮数做冒烟测试；
- 启动前运行 `verify_bundle.py`，校验关键文件和 SHA256；
- 日志写入调用者指定或包内新输出目录，不覆盖交付证据。

`evaluate_vx0p75.sh`：

- 固定 `vx=0.75 m/s、vy=0、yaw=0`；
- 单环境、确定性 policy；
- 记录原始视频、contact sheet 和逐步遥测；
- 不采样横移或转向命令；
- 评估退出后不残留 viewer 或训练进程。

## 8. ContactSensor 与指标

固定速度评估至少输出：

- 基座实际/目标 xy 速度和误差；
- yaw 角速度和误差；
- 左右脚 ContactSensor 原始接触布尔量、法向/合力（以运行时可用字段为准）；
- 左右脚 xy 速度和接触时滑移速度；
- 左右脚支撑期、摆动期、双支撑占比和步频；
- base height、roll/pitch/yaw、前倾角；
- episode step、存活时长、reset 和终止原因；
- 关节位置、速度、动作、限位余量；
- 自碰撞或非法接触指标（若任务暴露）。

`metrics_timeseries.npz` 保存逐步数值数组与名称；`metrics_summary.json` 保存单位、样本数、均值、分位数、最大值、存活和终止汇总；`contact_sensor_schema.json` 解释每个字段、shape、单位和来源。不可用字段必须明确写为 `unavailable` 并解释原因，不能编造。

## 9. 训练证据和曲线

保留原始 TensorBoard event 文件和能够找到的原始训练日志。通过 event 文件导出 `curves.csv` 与 `curves.png`，至少包含 reward、episode length、速度误差、前倾相关奖励/惩罚、滑移和终止指标中实际存在的项。`run_artifacts.json` 记录最终迭代、检查点、配置和文件哈希。

若原始文本日志不存在，README 必须明确说明，仅以 TensorBoard 为原始训练证据，不生成伪造日志。

## 10. 环境和 Docker

环境目录包含原 `pyproject.toml`、`uv.lock`、服务器 `.venv` 的 `pip freeze`、Python/CUDA/驱动/GPU/MuJoCo/MjLab/PyTorch 版本和依赖说明。

Dockerfile 以 NVIDIA CUDA 运行时为基础，复制包内源码并通过锁文件安装；README 明确需要 NVIDIA Container Toolkit、GPU 驱动兼容和 MuJoCo headless 渲染设置。Docker build 至少做语法/上下文检查；若因镜像网络或 GPU 条件未实际完成，必须如实记录，不能宣称验证通过。

## 11. 隐私和秘密净化

禁止进入压缩包：

- 服务器 IP、SSH host/port、密码、私钥、公钥配置和 known_hosts；
- API key、Token、cookie、`.env`、W&B 登录凭证；
- shell history、tmux 历史、交互式工具 上下文和聊天记录；
- 其他用户路径、进程信息和无关任务数据；
- 原服务器或本机绝对路径作为可执行依赖。

打包前后各运行一次内容与文件名扫描，包括 IPv4、常见密钥前缀、`password/token/secret` 赋值、SSH 文件名和已知敏感字符串。机器路径在配置中替换为 `${BUNDLE_ROOT}` 或包内相对路径；净化内容记录在 manifest，但绝不记录原秘密值。

## 12. 验收门禁

交付前必须通过：

1. 关键产物存在且非空；checkpoint、prior、norm 和动作数据加载成功。
2. 动作/关节/EE 顺序与模型和 policy 维度一致。
3. 固定 0.75 m/s 评估正常结束，JSON/NPZ 可解析且视频可解码。
4. TensorBoard 可读取，曲线 CSV/PNG 非空。
5. `SOURCE_MANIFEST.json` 覆盖包内除自身和 `SHA256SUMS.txt` 外的全部文件。
6. `SHA256SUMS.txt` 逐文件复算全部匹配。
7. 从压缩包解压到新目录后，`verify_bundle.py` 通过。
8. 两轮秘密扫描无服务器 IP、密码、Token 或 SSH 材料命中。
9. README 中没有依赖原服务器或本机绝对路径的步骤。

只有全部硬门禁通过才交付压缩包；环境条件限制项单独标为“未验证”，不冒充通过。
