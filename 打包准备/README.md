# HY2 打包准备说明

这个目录用于把 `/root/autodl-tmp/pipelines` 做成尽量自包含的可迁移目录。

## 1. 收集依赖到包内

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/打包准备/prepare_bundle_deps.sh
```

默认使用硬链接，把这些依赖收进：

```text
/root/autodl-tmp/pipelines/_bundle_deps
```

默认包含：

```text
upstream/     HY-World-2.0 和 MoGe 源码
conda-envs/   HY2 主管线环境和 vLLM 环境
```

模型权重默认不放进包里，因为 `模型下载/model_download.sh` 已经负责下载。

如果你确实想把模型也带进包里：

```bash
bash prepare_bundle_deps.sh --include-models
```

注意：不要在 `_bundle_deps` 里手动修改环境文件，因为硬链接和原文件共享同一个 inode。

## 2. 生成归档包

建议输出到外部盘或更大的磁盘：

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/打包准备/package_pipelines.sh \
  --out /path/to/external_disk/pipelines_hy2_bundle.tar
```

如果没有使用 `--include-models`，归档包不会包含模型权重。如果使用了 `--include-models`，归档逻辑体积可能超过 250G，建议输出到外部盘。

## 3. 迁移后运行

解压后目录结构应保留：

```text
pipelines/
  _bundle_deps/
  hy2_hyworld2_official/
```

如果包里没有模型，先运行：

```bash
bash pipelines/hy2_hyworld2_official/模型下载/model_download.sh
```

然后运行：

```bash
source pipelines/hy2_hyworld2_official/env.sh
source pipelines/hy2_hyworld2_official/activate_env.sh
python pipelines/hy2_hyworld2_official/python/preflight.py
```
