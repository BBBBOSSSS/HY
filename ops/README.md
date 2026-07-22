# ops/ — 运维与实验脚本

本目录存放 **tmux 运维、补跑、下载、环境修复** 等辅助脚本，**不是**日常主入口。

日常请用仓库根目录：

- `run_full_vlm_hy2.sh` / `run_all_hy2.sh` — 全链路
- `show_viewer.sh` — 查看 3DGS
- `env.sh` / `activate_env.sh` — 环境
- `stages/` — 分阶段运行

在仓库根执行示例：

```bash
bash ops/setup_in_tmux.sh
bash ops/resume_stage06_in_tmux.sh
```

脚本内 `ROOT` 指向仓库根目录（`ops/` 的上一级）。
