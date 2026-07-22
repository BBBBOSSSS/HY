# HY2.0 Official-Route Pipeline

This folder replaces the old MVGen / Matrix-style bridge. It uses the HY2.0 official world-generation route:

1. Optional panorama generation: use local `Qwen-Image-Edit-2509` base + `HY-Pano-2.0` LoRA by default; the full `HY-Pano-2.0` model remains optional.
2. WorldNav trajectory generation: `hyworld2/worldgen/traj_generate.py`.
3. Trajectory rendering: `traj_render.py`.
4. WorldStereo expansion: `video_gen.py` with `worldstereo-memory-dmd` by default (aligned with sh1_stable12k; non-DMD via `HY2_WORLDSTEREO_MODEL_TYPE=worldstereo-memory`).
5. GS data build: `gen_gs_data.py` (same `result_name` as Stage04 model type).
6. 3DGS training: defaults match **sh1_stable12k** (`sh_degree=1`, LPIPS 0.2/0.1, weaker densify, 12000 steps; see `scripts/env.sh`). Viewer: `show_gs.py`. MaskGaussian + anchor on; `HY2_CLEAN_PLY=0`.

This wrapper keeps the upstream WorldStereo default `--align-nframe 8`.
The trajectory length is still `--nframe 21`, matching the released
WorldStereo config. Stage 02, 03, 04, and 05 now validate camera/video frame
counts so a mismatched run cannot silently enter GS training.

The source checkout at `/root/autodl-tmp/upstream/HY-World-2.0` currently contains model/docs only. Scripts automatically fall back to `/root/autodl-tmp/upstream/HY-World-2.0-gh`, which contains `hyworld2/worldgen` code.

## Layout

See `FILES.md` for the file-by-file map.

```text
stages/   Stage launchers and full-chain entrypoints.
scripts/  Runtime shell helpers used by the pipeline: environment activation and vLLM.
python/   Python helpers used by stages and post-processing tools.
源码准备/  Reproduction-only upstream source checkout script.
conda环境配置/  Reproduction-only conda environment setup script.
模型下载/  Reproduction-only model download script.
打包准备/  Packaging helpers for collecting dependencies and creating archives.
../_bundle_deps/  Bundled upstream source and conda environments. Models are downloaded by script by default.
```

The repository root keeps only common entrypoints and source-able environment
wrappers. Stage launchers live in `stages/`, shell helpers live in `scripts/`,
and Python helpers live in `python/`.

## Reproduce From This Package

This package contains the pipeline wrappers, environment setup script, model
download script, source checkout script, and packaging helpers. If
`../_bundle_deps/` is present, the pipeline reads bundled source/envs first.
Models are downloaded by `模型下载/model_download.sh` by default. If bundled
source/envs are not present, run:

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/源码准备/setup_source_code.sh
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/conda环境配置/setup_conda_env.sh
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/模型下载/model_download.sh
```

Then verify:

```bash
source /root/autodl-tmp/pipelines/hy2_hyworld2_official/env.sh
source /root/autodl-tmp/pipelines/hy2_hyworld2_official/activate_env.sh
python /root/autodl-tmp/pipelines/hy2_hyworld2_official/python/preflight.py
```

To collect all currently installed dependencies into the package tree:

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/打包准备/prepare_bundle_deps.sh
```

## Preflight

```bash
source /root/autodl-tmp/pipelines/hy2_hyworld2_official/env.sh
source /root/autodl-tmp/pipelines/hy2_hyworld2_official/activate_env.sh
python /root/autodl-tmp/pipelines/hy2_hyworld2_official/python/preflight.py
```

The venv is already the intended runtime path: `torch==2.7.1+cu128` and
`torchvision==0.22.1+cu128`. Avoid running the pipeline with system Python,
which currently has an older CUDA build.

The checked-in `gsplat` CUDA extension is built for `sm_89`. That is correct
for RTX 6000 Ada-class cards. If preflight reports a different compute
capability, rebuild it on the GPU machine before GS training:

```bash
cd /root/autodl-tmp/upstream/HY-World-2.0-gh/hyworld2/worldgen/third_party/gsplat_maskgaussian
TORCH_CUDA_ARCH_LIST="8.9" /root/autodl-tmp/.venv/bin/pip install -e . --no-build-isolation
```

For a Blackwell Pro 6000, use the capability reported by preflight, for example
`TORCH_CUDA_ARCH_LIST="12.0"`, and make sure `CUDA_HOME` points to a CUDA toolkit
new enough for that architecture.

The default panorama route now expects the lightweight local assets:

```text
/root/autodl-tmp/models/Qwen-Image-Edit-2509
/root/autodl-tmp/models/HY-World-2.0-modelscope-full/HY-Pano-2.0/pytorch_lora_weights.safetensors
```

Prepare these model files before running the pipeline.

For a reproducible download entrypoint:

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/模型下载/model_download.sh
```

For a reproducible environment setup entrypoint:

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/conda环境配置/setup_conda_env.sh
```

The wrapper can still auto-detect a full local `HY-Pano-2.0` install at:

```text
/root/autodl-tmp/models/HY-World-2.0-modelscope-full/HY-Pano-2.0
```

If that path is present and you explicitly set `HY_PANO_BACKEND=hy-pano2`,
`run_01_panorama.sh` uses it. The default path is now `HY_PANO_BACKEND=qwen-lora`.

For the full `HY-Pano-2.0` model, the wrapper also supports single-process
multi-GPU loading through `transformers + accelerate` device maps:

```bash
export HY_PANO_BACKEND=hy-pano2
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HY_PANO_DEVICE_MAP=auto
export HY_PANO_MAX_GPU_MEMORY=70GiB
export HY_PANO_MAX_CPU_MEMORY=128GiB
```

`HY_PANO_MAX_GPU_MEMORY` can also be set per device, for example
`0:70GiB,1:70GiB,2:70GiB,3:70GiB`.

MoGe and Uni3C are expected to be uploaded manually:

```text
/root/autodl-tmp/models/moge-2-vitl-normal/model.pt
/root/autodl-tmp/models/Uni3C/controlnet.pth
```

## Run From An Existing Panorama

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/run_full_vlm_hy2.sh \
  --name hy2_test \
  --panorama /path/to/panorama.png \
  --scene-type outdoor \
  --nproc 1 \
  --steps 12000
```

## Run From A Perspective Image

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/run_full_vlm_hy2.sh \
  --name hy2_img_test \
  --image /root/autodl-tmp/6a3bf4c4c95bd2c17ec3339bd38803d6.jpg \
  --prompt "bright warm cozy modern studio bedroom, clean natural lighting, realistic layout" \
  --scene-type indoor \
  --nproc 1 \
  --steps 12000
```

## Full VLM Mode

Use the full VLM wrapper for one-command production runs. It starts the Qwen3-VL
vLLM server, waits for `/v1/models`, keeps vLLM alive through Stage03 captions,
then stops the tmux session before Stage04.

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/run_full_vlm_hy2.sh \
  --name RUN_NAME \
  --image /path/to/input.jpg \
  --scene-type outdoor \
  --nproc 1 \
  --steps 12000
```

`run_all_hy2.sh` remains the lower-level stage chain. For tests that intentionally
avoid vLLM, call it with `--mode offline-basic`.

## Viewer

```bash
bash /root/autodl-tmp/pipelines/hy2_hyworld2_official/show_viewer.sh --name hy2_test --port 7007
```
