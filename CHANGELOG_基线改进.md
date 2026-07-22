# HY-World 2.0 基线：本轮改进与修改说明

本文记录本仓库相对初始 HY2 管线封装的**最新一轮基线调整**（不含情感数据集相关代码）。  
目标：在官方 image-to-world 路线上，用更稳妥的默认配置提升 3DGS 重建可用性，并理清仓库目录。

---

## 1. 背景

在完整跑通「全景 → WorldNav → WorldStereo → GS 数据 → 3DGS 训练 → Viewer」后，发现：

- 默认训练配置下重建观感与验证指标仍有提升空间；
- 盲目提高球谐阶数 / LPIPS / densify 反而变差；
- 仓库根目录堆叠大量 tmux 运维脚本，GitHub 浏览杂乱。

本轮在**同一测试全景**上做了多组对照，并将最优配方固化为默认。

---

## 2. 核心结论（请直接采用）

| 项目 | 默认选择 | 说明 |
|------|----------|------|
| WorldStereo | **worldstereo-memory-dmd** | 本场景下 non-DMD memory 无明显提升，默认仍用 DMD |
| 3DGS 训练 | **sh1_stable12k** | 当前验证最优配方（见下表） |
| 仓库布局 | 根目录仅主入口，运维进 **ops/** | 便于阅读与同步 |

### 同场景验证对比（图书馆室内全景）

| 配置 | 多视角 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 约高斯数 | 结论 |
|------|--------|--------|--------|---------|----------|------|
| 早期默认 | DMD | 23.8 | 0.815 | 0.191 | ~1087 万 | 可用基线 |
| sh2 + 高 LPIPS + 8k | 同 DMD 数据 | 20.9 | 0.750 | 0.264 | ~1540 万 | **不采用** |
| **sh1_stable12k** | **DMD** | **26.5** | **0.869** | **0.134** | **~415 万** | **默认** |
| memory + 同 sh1 稳训 | memory | 25.8 | 0.866 | 0.135 | ~413 万 | 相对最优无增益 |

---

## 3. sh1_stable12k 训练策略（已写入默认）

### 3.1 参数一览

| 项 | 值 | 相对早期常见设置 |
|----|-----|------------------|
| `sh_degree` | **1** | 原多为 0（全程 DC） |
| `sh_degree_interval` | **1500** | 阶梯 0→1（需打上游补丁） |
| `lpips_lambda1` / `lpips_lambda2` | **0.2 / 0.1** | 保持官方默认，不盲目加大 |
| `max_steps` | **12000** | 常见 8000 → 加长 |
| densify `grow_grad2d` | **0.00025** | 原 0.0001，更难增生 |
| densify `refine_every` | **150** | 原 100 |
| densify 停止 | 约 **40% × steps**（12k 时 ~4800） | 原约 50% 仍 densify |
| MaskGaussian / anchor | **开** | 与官方 README 对齐 |
| opacity 过滤导出 | **默认关** | 避免镂空/变糊 |
| 2× 固定超分导出 | **默认关** | 需要时再开 |

环境变量（`scripts/env.sh`）：

```bash
HY2_WORLDSTEREO_MODEL_TYPE=worldstereo-memory-dmd   # 默认
HY2_SH_DEGREE=1
HY2_SH_DEGREE_INTERVAL=1500
HY2_LPIPS_LAMBDA1=0.2
HY2_LPIPS_LAMBDA2=0.1
HY2_GROW_GRAD2D=0.00025
HY2_REFINE_EVERY=150
HY2_REFINE_STOP_FRAC=0.40
HY2_GS_STEPS=12000
HY2_EXPORT_RENDER=0
HY2_CLEAN_PLY=0
HY2_USE_MASK_GAUSSIAN=1
HY2_USE_ANCHOR_PROTECTION=1
```

Stage 脚本：`stages/run_04_worldstereo.sh`、`run_05_gs_data.sh`、`run_06_train_gs.sh` 已对齐上述默认。

### 3.2 相对前几轮的主要改进点

1. **sh0 → sh1**  
   适度视角相关外观，比全程 DC 更稳；避免 sh2 难训、易糊、易爆点数。

2. **8k → 12k**  
   densify 提前停后，后半程继续拟合，PSNR/SSIM 上升、LPIPS 下降。

3. **减弱 densify**  
   高斯数从约千万级压到约 **400 万**，过拟合与「糊片」更少，Viewer 更轻。

4. **LPIPS 不盲目加大**  
   高 LPIPS 在生成式低清监督上易「抹平」，本轮保持 0.2/0.1。

5. **多视角默认 DMD**  
   同 GS 配方下 memory 与 DMD 接近，DMD 作为默认更省事；需要时再切 `worldstereo-memory`。

6. **不做 2× 强行导出当默认**  
   避免把软重建放大误当作画质提升。

---

## 4. 上游补丁：SH 阶梯升阶

官方/部分检出中 `world_gs_trainer.py` 曾**写死** `sh_degree_to_use = cfg.sh_degree`（不爬坡）。  
本仓提供补丁，在 `sh_degree > 0` 时恢复：

```text
sh_degree_to_use = min(step // sh_degree_interval, sh_degree)
```

位置：

```text
patches/world_gs_trainer_enable_sh_schedule.patch
patches/README.md
```

在 **HY-World 上游源码根目录** 应用：

```bash
cd /path/to/HY-World-2.0-gh
patch -p1 < /path/to/HY/patches/world_gs_trainer_enable_sh_schedule.patch
```

与 Stage06 的 `HY2_SH_DEGREE=1`、`HY2_SH_DEGREE_INTERVAL=1500` 配合使用。

---

## 5. 仓库结构整理

### 5.1 根目录只保留主入口

- `env.sh` / `activate_env.sh`
- `run_full_vlm_hy2.sh` / `run_all_hy2.sh`
- `show_viewer.sh`
- `stages/`、`scripts/`、`python/`
- `conda环境配置/`、`模型下载/`、`源码准备/`、`打包准备/`
- `patches/`、`README.md`、`FILES.md`

### 5.2 运维脚本迁入 `ops/`

以下类型脚本已移出根目录，避免 GitHub 文件列表杂乱：

- tmux 环境安装 / 修复  
- 模型下载辅助  
- Stage05/06 断点续跑  
- 单场景实验与 GPU 诊断  

用法示例：

```bash
bash ops/setup_in_tmux.sh
bash ops/resume_stage06_in_tmux.sh
```

说明见 `ops/README.md`、`FILES.md`。

### 5.3 不同步内容

- 环境本体：`_bundle_deps/`（conda、上游大树缓存）  
- 模型权重：`models/`、`*.safetensors` 等  
- 运行产出：`outputs/`、`logs/`  
- 情感数据集管线目录（若本地存在）不纳入本基线同步范围  

---

## 6. 涉及的主要文件

| 路径 | 修改要点 |
|------|----------|
| `scripts/env.sh` | DMD + sh1_stable12k 默认环境变量 |
| `stages/run_04_worldstereo.sh` | 默认 `worldstereo-memory-dmd` |
| `stages/run_05_gs_data.sh` | 与 Stage04 的 result_name 默认一致 |
| `stages/run_06_train_gs.sh` | sh/LPIPS/densify/步数/可选固定分辨率导出 |
| `python/export_gs_fixed_res.py` | 可选：按训练分辨率×scale 导出预览图 |
| `python/render_long_gs_tours.py` | 导出分辨率 / sh 默认与环境对齐 |
| `patches/*` | 上游 SH 阶梯补丁 |
| `ops/*` | 运维脚本集中存放 |
| `README.md` / `FILES.md` | 结构与默认配置说明 |

---

## 7. 使用建议

1. 新机器：按 `源码准备` → `conda环境配置` → `模型下载` 装依赖，再 `patch` 上游 SH 阶梯。  
2. 全链路：`bash run_full_vlm_hy2.sh ...`（自动用当前默认）。  
3. 仅重训 GS：`bash stages/run_06_train_gs.sh --name <RUN> --steps 12000`。  
4. 预览：`bash show_viewer.sh --name <RUN> --port 6008`。  
5. 若要试 memory 多视角：`export HY2_WORLDSTEREO_MODEL_TYPE=worldstereo-memory` 后再跑 Stage04–06。

---

## 8. 版本记录（本说明对应改动）

- 固化 **sh1_stable12k + DMD** 为基线默认。  
- 增加上游 **SH 阶梯** 补丁说明与文件。  
- 仓库 **ops/** 整理，根目录减负。  

*文档随基线策略更新；实验数字来自固定测试全景上的验证集统计，换场景请以本地 val 为准。*
