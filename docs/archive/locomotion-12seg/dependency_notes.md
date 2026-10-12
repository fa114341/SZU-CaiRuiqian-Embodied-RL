<!-- Public, sanitized copy of a historical working document. Redacted lines or fields: 0. Do not use as a complete reproduction guide. -->

# 训练环境记录

- 操作系统：Linux 服务器
- Python：3.13.15
- GPU：NVIDIA RTX 5090（32607 MiB）
- NVIDIA Driver：580.159.04
- PyTorch：2.10.0+cu128
- NumPy：2.5.2
- MuJoCo：3.11.1.dev964882123
- mjlab：Git revision `731fa27b8c023d3d5258501c6949779425d2a37fd`
- mujoco-warp：Git revision `88b55fc2696960bb927bc12584994bb8412b36558`
- rsl-rl-lib：5.2.0
- tensorboard：2.21.0
- tyro：1.0.15
- wandb：0.28.2
- warp-lang：1.16.0

精确 Python 依赖与索引配置以 `source/smp/uv.lock` 和
`source/smp/pyproject.toml` 为准。本简化包不额外构建 Docker 镜像。
