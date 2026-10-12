<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# T800 action、joint 与 EE 顺序

策略 action 与关节顺序均为 25 维，严格按 `joint_order.json` 中数组顺序。
动作值通过同文件记录的 scale 转成关节位置目标。

SMP 特征为 55 维：`root_pos(3) + root_rot_6d(6) + joint_pos(25) +
ee_pos(5×3) + root_lin_vel(3) + root_ang_vel(3)`。

EE 顺序：左踝、右踝、`LINK_BASE`、左前臂末端、右前臂末端。MuJoCo 会把
fixed wrist/foot link 合并到相邻 body，因此这里使用踝 roll 和肘 yaw link 代理。

映射说明：

- `J24_HEAD_YAW` 是头部 yaw 关节；`LINK_HEAD_YAW` 是头部 link，不在本次 5 个 EE 中。
- `J12_TORSO_YAW` 是躯干 yaw 关节；`LINK_WAIST_YAW` 是腰部/躯干 body，环境中可作 viewer/body-COM 锚点，不是头部，也不在本次 EE 中。
- 本次 SMP 的躯干/root EE 使用 `LINK_BASE`。
- 历史 HOME 配置中的 `J13_SHOULDER_ROLL_L` 未命中真实关节；真实左肩 roll 为 J14。该事实原样保留。
