# HY2 管线文件说明

这个目录现在按职责拆开。根目录只保留最常用入口，真正实现放在子目录里。

```text
stages/   阶段启动文件和全链路入口
scripts/  管线运行时会实际调用的环境和 vLLM shell 工具
python/   阶段脚本调用的 Python 实现和检查工具
ops/      tmux 运维、补跑、下载、环境修复（非日常主入口）
patches/  上游 HY-World 补丁（如 SH 阶梯）
源码准备/  复现项目用的一键上游源码下载/更新脚本
conda环境配置/  复现项目用的一键 conda 环境配置脚本
模型下载/  复现项目用的一键模型下载脚本，不参与运行时链路
打包准备/  把外部依赖硬链接收进 pipelines/_bundle_deps，并生成归档包
../_bundle_deps/  打包时收进去的上游源码和 conda 环境依赖；模型默认不放入包内
```

## 根目录文件

这些文件有用，建议保留。它们是为了日常运行方便，不用每次都进子目录。

```text
env.sh
  source 入口。加载 scripts/env.sh，设置 HY2_PIPELINE_ROOT、HY2_STAGE_ROOT、
  HY2_SCRIPT_ROOT、HY2_PY_ROOT、模型路径、输出路径和默认运行参数。

activate_env.sh
  source 入口。加载 scripts/activate_env.sh，激活 HY2 Python/CUDA 环境。

run_full_vlm_hy2.sh
  推荐的一键满血入口。自动启动 vLLM，等待服务可用，跑 full VLM 的 Stage02/03，
  Stage03 caption 完成后杀掉 vLLM，然后继续 Stage04-06。

run_all_hy2.sh
  低层全链路入口。可以串 Stage00-06，但不负责启动或管理 vLLM。
  主要用于 offline-basic 测试或手动调度。

show_viewer.sh
  viewer 入口。按 run 名找到最新 checkpoint，启动官方 HY-World GS 查看器。

README.md
  运行说明和快速入口。

FILES.md
  当前文件说明，也就是本文档。
```


## ops/

```text
setup_in_tmux.sh / repair_environment_in_tmux.sh
  环境安装与修复（在 tmux 中跑）。

download_*_in_tmux.sh
  模型下载辅助。

resume_* / rerun_*_in_tmux.sh
  断点续跑 Stage05/06 等。

run_dmd_library_full_in_tmux.sh / test_image_in_tmux.sh / diagnose_vllm_gpu_in_tmux.sh
  场景实验与诊断。
```

## stages/

```text
run_00_prepare_scene.sh
  Stage00。准备场景输入，复制图片或校验 2:1 全景图，并写 meta_info.json。

run_01_panorama.sh
  Stage01。用 Qwen-Image-Edit + HY-Pano LoRA 生成候选全景图并选优。

run_02_worldnav.sh
  Stage02。运行 WorldNav：深度、视角拆分、导航目标、camera.json 轨迹规划。

run_03_traj_render.sh
  Stage03。渲染轨迹 render.mp4，并生成 traj_caption.json。

run_03caption_04_06.sh
  补跑入口。已有 Stage03 render.mp4 时，重新补 caption，然后继续 Stage04-06。

run_04_worldstereo.sh
  Stage04。运行 WorldStereo / DMD，生成扩展视频。

run_05_gs_data.sh
  Stage05。把 WorldStereo 输出整理成 3DGS 训练数据。

run_06_train_gs.sh
  Stage06。训练 3DGS，导出 PLY/SPZ，并做 opacity 清理和 viewer 对齐。

run_full_vlm_hy2.sh
  满血一键入口的真实实现。

run_all_hy2.sh
  低层全链路入口的真实实现。

show_viewer.sh
  viewer 入口的真实实现。
```

## scripts/

```text
env.sh
  统一环境变量定义。

activate_env.sh
  激活 HY2 运行环境。

start_vllm_qwen3vl.sh
  在 tmux 里启动 Qwen3-VL vLLM 服务。
```

## python/

```text
prepare_scene.py
  Stage00 实现。

generate_and_select_panorama.py
  Stage01 实现，包含全景生成、候选评分和选优。

check_camera_json_counts.py
  检查 camera.json 帧数。

check_frame_camera_counts.py
  检查 WorldStereo 结果视频帧数是否和 camera.json 对得上。

preflight.py
  预检模型、代码路径、CUDA、gsplat 等是否齐全。

require_cuda.py
  CUDA 阶段的早期 GPU 检查。

recaption_traj_videos.py
  用 Qwen3-VL 给已有 render.mp4 重新写 traj_caption.json。

clean_gaussian_ply_opacity.py
  按 opacity 清理导出的 Gaussian PLY。

align_gaussian_ply_for_viewer.py
  把导出的 Gaussian PLY 对齐到浏览器 viewer 需要的方向。
```
