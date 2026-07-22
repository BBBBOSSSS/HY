# HY2 conda 环境配置说明

这个目录只放复现项目用的环境配置脚本，不参与管线运行时逻辑。

## 一键配置

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/conda环境配置/setup_conda_env.sh
```

默认会创建两个环境：

```text
/root/autodl-tmp/pipelines/_bundle_deps/conda-envs/hyworld2
  HY2 主管线环境：Stage00-06、WorldNav、WorldStereo、3DGS 训练等。

/root/autodl-tmp/pipelines/_bundle_deps/conda-envs/vllm_qwen
  Qwen3-VL 的 vLLM 服务环境。
```

默认 PyTorch 栈按当前项目可运行环境配置：

```text
torch==2.7.1
torchvision==0.22.1
torchaudio==2.7.1
CUDA wheel 源：https://download.pytorch.org/whl/cu128
vllm==0.23.0
```

## 常用参数

```bash
# 只配置主管线环境，不配置 vLLM
bash setup_conda_env.sh --skip-vllm

# 不重编译 gsplat
bash setup_conda_env.sh --skip-gsplat

# 只安装环境，不跑 preflight
bash setup_conda_env.sh --skip-preflight

# 环境已存在时强制删除重建
bash setup_conda_env.sh --force

# 只看安装计划，不真正执行
bash setup_conda_env.sh --dry-run
```

## 切换 PyTorch/CUDA wheel 源

如果目标机器不是 CUDA13，可覆盖 PyTorch 源和版本：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
TORCH_PACKAGES="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1" \
bash setup_conda_env.sh
```

## 推荐复现顺序

```bash
# 1. 准备上游源码
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/源码准备/setup_source_code.sh

# 2. 配环境
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/conda环境配置/setup_conda_env.sh

# 3. 下载模型
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/模型下载/model_download.sh

# 4. 跑满血管线
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/run_full_vlm_hy2.sh \
  --name demo \
  --panorama /path/to/pano.png \
  --scene-type outdoor
```

环境配置完成后，手动激活方式：

```bash
source /root/autodl-tmp/pipelines/hy2_hyworld2_official/env.sh
source /root/autodl-tmp/pipelines/hy2_hyworld2_official/activate_env.sh
```
