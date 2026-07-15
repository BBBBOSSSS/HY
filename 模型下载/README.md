# HY2 模型下载说明

这个目录只放复现项目用的模型下载脚本，不参与管线运行时逻辑。

## 一键下载

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/模型下载/model_download.sh
```

默认下载到：

```text
/root/autodl-tmp/pipelines/_bundle_deps/models
```

这个路径和管线里的 `scripts/env.sh` 默认模型路径一致，下载完不需要再改环境变量。

## 默认来源

```text
ModelScope:
  Tencent-Hunyuan/HY-World-2.0
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/HY-World-2.0-modelscope-full

  Qwen/Qwen-Image-Edit-2509
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/Qwen-Image-Edit-2509

  Qwen/Qwen3-VL-8B-Instruct
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/Qwen3-VL-8B-Instruct

  Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/Wan2.1-I2V-14B-720P-Diffusers

  facebook/sam3
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/sam3

  facebook/dinov2-base
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/dinov2-base

  IDEA-Research/grounding-dino-tiny
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/grounding-dino-tiny

  bluestone25/moge-vitl
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/moge-2-vitl-normal

Hugging Face 协议，默认走国内镜像 HF_ENDPOINT=https://hf-mirror.com:
  hanshanxue/WorldStereo
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/worldstereo

  ewrfcas/Uni3C
    -> /root/autodl-tmp/pipelines/_bundle_deps/models/Uni3C
```

## 常用参数

```bash
# 只下载 WorldStereo 和 Uni3C
bash model_download.sh --only worldstereo,uni3c

# 指定模型根目录
bash model_download.sh --model-root /root/autodl-tmp/pipelines/_bundle_deps/models

# 切回 Hugging Face 官方源
bash model_download.sh --hf-endpoint https://huggingface.co --only worldstereo,uni3c

# 只看下载计划
bash model_download.sh --dry-run

# 下载完整 HY-Pano-2.0，而不是只下载 LoRA
bash model_download.sh --full-hy-pano
```

## 可覆盖变量

如果以后你把某个权重搬到自己的 ModelScope 或 Hugging Face 仓库，只需要覆盖 repo id：

```bash
export WORLDSTEREO_HF_REPO_ID=your-org/WorldStereo
export UNI3C_HF_REPO_ID=your-org/Uni3C
bash model_download.sh --only worldstereo,uni3c
```

下载结束后脚本会自动跑管线 `preflight.py`。如果只想下载不预检：

```bash
bash model_download.sh --skip-preflight
```
