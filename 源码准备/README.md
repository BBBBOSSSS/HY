# HY2 源码准备说明

这个目录只放复现项目用的上游源码准备脚本，不参与管线运行时逻辑。

## 一键下载/更新上游源码

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/源码准备/setup_source_code.sh
```

默认会把官方源码放到：

```text
/root/autodl-tmp/pipelines/_bundle_deps/upstream/HY-World-2.0-gh
```

这个路径和管线 `scripts/env.sh` 默认读取路径一致。

## 国内网络

如果 GitHub 较慢，可以把仓库换成自己的镜像：

```bash
HY2_CODE_REPO_URL=https://你的镜像/HY-World-2.0.git \
bash setup_source_code.sh
```

## 推荐复现顺序

```bash
# 1. 准备上游源码
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/源码准备/setup_source_code.sh

# 2. 配置 conda 环境
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/conda环境配置/setup_conda_env.sh

# 3. 下载模型
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/模型下载/model_download.sh

# 4. 运行管线
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/run_full_vlm_hy2.sh --help
```
