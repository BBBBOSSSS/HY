#!/usr/bin/env python3
# 用途：阶段 01 工具；生成全景候选并选择最佳结果。
import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import utils3d
from PIL import Image

if os.environ.get("MOGE_LEGACY", "0") == "1":
    from moge.model.v1 import MoGeModel
else:
    from moge.model.v2 import MoGeModel

from src.panorama_utils import pred_pano_depth, rotate_around_z_axis, split_panorama_depth, split_panorama_image

DIFFUSION_WIDTH = 1952
TARGET_WIDTH = 1920
TARGET_HEIGHT = 960
BLEND_WIDTH = DIFFUSION_WIDTH - TARGET_WIDTH
DEFAULT_HY_PANO_MODEL = "/root/autodl-tmp/models/HY-World-2.0-modelscope-full/HY-Pano-2.0"
DEFAULT_QWEN_PANO_BASE = "/root/autodl-tmp/models/Qwen-Image-Edit-2509"
DEFAULT_HY_PANO_LORA_REPO = "/root/autodl-tmp/models/HY-World-2.0-modelscope-full"
DEFAULT_HY_PANO_OFFLOAD_DIR = "/root/autodl-tmp/outputs/hy2_worldgen_runs/_hf_offload/hy_pano2"
_HY_PANO_NEGATIVE_PROMPT_WARNED = False

QWEN_POSITIVE_PREFIX = (
    "Create a **ERP** panoramic expansion of the provided image. "
    "Preserve the original style, lighting, and fine details seamlessly "
    "throughout the extended areas, extend according to: "
)
QWEN_POSITIVE_SUFFIX = " 8k UHD, masterpiece, razor-sharp details."
QWEN_NEGATIVE_PREFIX = (
    "低分辨率，低画质，模糊。杂乱的背景，结构扭曲，模糊纹理，物体融合。构图混乱。"
    "过度光滑，画面具有AI感。人脸畸形。巨大物体，巨大建筑，近景特写，近景压迫，比例失调。"
    "车，车辆。画面上方的树叶。"
)


DEFAULT_INDOOR_PROMPT = (
    "Expand this image into a bright warm coherent 360-degree equirectangular indoor panorama."
)

DEFAULT_OUTDOOR_PROMPT = (
    "Expand this image into a compact coherent outdoor 360-degree equirectangular panorama with a controlled courtyard-scale layout."
)

INDOOR_GEOMETRY_SUFFIX = (
    " Preserve a single coherent room-scale 360-degree scene. The rear side and all unseen back-side areas must be sharp, "
    "fully detailed, spatially consistent, and navigable. Keep stable perspective, continuous floor geometry, consistent "
    "materials, lighting, scale, and object identity. Avoid vague backgrounds, low-detail rear regions, repeated objects, "
    "broken architecture, warped structures, and smeared textures."
)

OUTDOOR_COMPACT_SUFFIX = (
    " Preserve the original subject identity and expand only into a compact walkable outdoor courtyard or garden-scale scene, "
    "not a vast plaza or distant landscape. Keep important structures, plants, walls, railings, steps, ground paving, and "
    "landmarks in the near-to-mid range with human-scale depth. The 360-degree rear side must be sharp, fully detailed, "
    "spatially consistent, and navigable, with continuous ground and stable geometry. Keep the layout enclosed or semi-enclosed "
    "with nearby boundaries and clear parallax cues. Avoid empty open space, far-away tiny architecture, "
    "large blank pavement, vague horizon backgrounds, wide barren plazas, repeated objects, broken roofs, warped walls, "
    "smeared trees, and texture stretching."
)

DEFAULT_NEGATIVE_PROMPT = (
    "fisheye, warped room, curved walls, bent floor, duplicated objects, repeated objects, floating objects, "
    "twisted geometry, collage artifacts, broken perspective, mirror seams, blur, haze, fog, "
    "overexposed highlights, black voids, heavy distortion, stretched textures, malformed architecture, "
    "vast empty plaza, huge empty pavement, distant skyline, far-away tiny buildings, vague horizon, "
    "open barren landscape, low-detail rear side, smeared trees"
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _force_panorama_size(image: Image.Image) -> Image.Image:
    if image.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return image
    return image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)


def _circular_blend_edges(image: Image.Image, blend_width: int = BLEND_WIDTH) -> Image.Image:
    """Blend the generated overlap into the left ERP edge, then remove it."""
    arr = np.array(image.convert("RGB"), dtype=np.uint8).copy()
    if blend_width <= 0 or blend_width >= arr.shape[1]:
        raise ValueError(f"Invalid panorama blend width {blend_width} for image width {arr.shape[1]}")
    for x in range(blend_width):
        alpha = x / blend_width
        arr[:, x, :] = arr[:, -blend_width + x, :] * (1 - alpha) + arr[:, x, :] * alpha
    return Image.fromarray(arr[:, :-blend_width].astype(np.uint8), mode="RGB")


def _finalize_panorama(image: Image.Image) -> Image.Image:
    if image.size != (DIFFUSION_WIDTH, TARGET_HEIGHT):
        image = image.resize((DIFFUSION_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    return _force_panorama_size(_circular_blend_edges(image))


def _is_current_panorama(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            return image.size == (TARGET_WIDTH, TARGET_HEIGHT)
    except (OSError, ValueError):
        return False


def _prepare_prompt(prompt: str, scene_type: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        prompt = DEFAULT_OUTDOOR_PROMPT if scene_type == "outdoor" else DEFAULT_INDOOR_PROMPT

    lower = prompt.lower()
    if scene_type == "outdoor":
        if "compact walkable outdoor" in lower or "near-to-mid range" in lower:
            return prompt
        return f"{prompt.rstrip('. ')}.{OUTDOOR_COMPACT_SUFFIX}"

    if "straight walls" in lower and "coherent room" in lower:
        return prompt
    return f"{prompt.rstrip('. ')}.{INDOOR_GEOMETRY_SUFFIX}"


def _laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _brightness_score(brightness: float) -> float:
    return _clamp01(1.0 - abs(brightness - 0.58) / 0.36)


def _contrast_score(contrast: float) -> float:
    return _clamp01(contrast / 0.22)


def _sharpness_score(sharpness: float) -> float:
    # Panorama candidates often land in the 300-2500 Laplacian range. The old
    # log/5.6 score saturated too early, treating visibly soft 700-ish outputs
    # as equivalent to crisp 1900+ outputs.
    low = math.log1p(350.0)
    high = math.log1p(2400.0)
    return _clamp01((math.log1p(max(sharpness, 0.0)) - low) / (high - low))


def _build_mid_ring_cameras(rot_deg: float = 30.0, h: int = 416, w: int = 416):
    start_point = np.array([-1, 0, 0], dtype=np.float32)
    direct_points = [start_point]
    n_view = int(round(360.0 / rot_deg))
    for i in range(1, n_view):
        direct_points.append(rotate_around_z_axis(start_point.reshape(1, 3), rot_deg * i)[0])
    direct_points = np.stack(direct_points, axis=0)
    intrinsics = utils3d.numpy.intrinsics_from_fov(fov_x=np.deg2rad(120.0), fov_y=np.deg2rad(90.0))
    extrinsics = utils3d.numpy.extrinsics_look_at(np.array([0, 0, 0]), direct_points, np.array([0, 0, 1])).astype(np.float32)
    intrinsics = [intrinsics] * len(direct_points)
    return extrinsics, intrinsics, h, w


def _hy_pano_model_ready(model_path: Path) -> bool:
    return model_path.is_dir() and (model_path / "config.json").exists() and (model_path / "model.safetensors.index.json").exists()


def _qwen_pano_required_paths(base_path: Path) -> list[Path]:
    return [
        base_path / "model_index.json",
        base_path / "configuration.json",
        base_path / "transformer" / "config.json",
        base_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json",
        base_path / "text_encoder" / "config.json",
        base_path / "text_encoder" / "model.safetensors.index.json",
        base_path / "vae" / "config.json",
        base_path / "vae" / "diffusion_pytorch_model.safetensors",
        base_path / "processor" / "tokenizer.json",
    ]


def _hy_pano_qwen_ready(base_path: Path, lora_repo: Path, lora_subfolder: str) -> bool:
    required_ok = all(path.exists() for path in _qwen_pano_required_paths(base_path))
    lora_ok = (lora_repo / lora_subfolder / "pytorch_lora_weights.safetensors").exists()
    return required_ok and lora_ok


def _normalize_device_map(device_map: str | None):
    if device_map is None:
        return None
    normalized = str(device_map).strip()
    if not normalized or normalized.lower() in {"none", "single", "disabled"}:
        return None
    if normalized.isdigit():
        return int(normalized)
    return normalized


def _parse_max_memory(spec: str | None, gpu_count: int):
    if spec is None:
        return None
    spec = spec.strip()
    if not spec:
        return None

    entries = [entry.strip() for entry in spec.split(",") if entry.strip()]
    if not entries:
        return None

    has_mapping = all(":" in entry for entry in entries)
    if not has_mapping:
        return {idx: spec for idx in range(gpu_count)}

    parsed = {}
    for entry in entries:
        key, value = entry.split(":", 1)
        key = key.strip()
        value = value.strip()
        parsed[int(key) if key.isdigit() else key] = value
    return parsed


def _build_max_memory(max_gpu_memory: str | None, max_cpu_memory: str | None):
    gpu_count = torch.cuda.device_count()
    max_memory = _parse_max_memory(max_gpu_memory, gpu_count)
    if max_cpu_memory and max_cpu_memory.strip():
        if max_memory is None:
            max_memory = {}
        max_memory["cpu"] = max_cpu_memory.strip()
    return max_memory or None


def _summarize_device_map(model) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return "single-device"
    counts = {}
    for target in device_map.values():
        key = str(target)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _select_backend(args) -> str:
    full_model_path = Path(args.hy_pano_model)
    qwen_base = Path(args.model_id)
    lora_repo = Path(args.lora_repo)

    if args.backend == "auto":
        if _hy_pano_model_ready(full_model_path):
            return "hy-pano2"
        if _hy_pano_qwen_ready(qwen_base, lora_repo, args.lora_subfolder):
            return "qwen-lora"
        raise FileNotFoundError(
            "No usable panorama backend found. Missing local HY-Pano-2.0 full model and missing Qwen-Image-Edit + HY-Pano LoRA."
        )

    if args.backend == "hy-pano2" and not _hy_pano_model_ready(full_model_path):
        raise FileNotFoundError(f"HY-Pano-2.0 full model is incomplete or missing: {full_model_path}")
    if args.backend == "qwen-lora" and not _hy_pano_qwen_ready(qwen_base, lora_repo, args.lora_subfolder):
        missing = [str(path) for path in _qwen_pano_required_paths(qwen_base) if not path.exists()]
        lora_weight = lora_repo / args.lora_subfolder / "pytorch_lora_weights.safetensors"
        if not lora_weight.exists():
            missing.append(str(lora_weight))
        raise FileNotFoundError(
            "Qwen panorama backend is incomplete. Missing required files: " + ", ".join(missing)
        )
    return args.backend


def _load_hy_pano_modules(model_dir: Path):
    import transformers.utils as tf_utils
    import transformers.utils.import_utils as tf_import_utils

    sys.modules.setdefault("utils", tf_utils)
    sys.modules.setdefault("utils.import_utils", tf_import_utils)

    package_name = "_hy_pano2_local_pkg"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            model_dir / "__init__.py",
            submodule_search_locations=[str(model_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create import spec for {model_dir}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    modeling_mod = importlib.import_module(f"{package_name}.modeling_hunyuan_image_3")
    config_mod = importlib.import_module(f"{package_name}.configuration_hunyuan_image_3")
    return modeling_mod, config_mod


def _load_hy_pano_model(
    model_dir: Path,
    dtype: torch.dtype,
    device: str,
    device_map: str | int | None,
    max_memory: dict | None,
    offload_dir: Path,
):
    from transformers import GenerationConfig

    modeling_mod, config_mod = _load_hy_pano_modules(model_dir)
    config = config_mod.HunyuanImage3Config.from_pretrained(str(model_dir), local_files_only=True)
    if not hasattr(config, "model_version"):
        config.model_version = "hunyuan-image-3"
    load_kwargs = dict(
        config=config,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    )
    if device_map is not None:
        offload_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs["device_map"] = device_map
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        load_kwargs["offload_folder"] = str(offload_dir)

    model = modeling_mod.HunyuanImage3ForCausalMM.from_pretrained(
        str(model_dir),
        **load_kwargs,
    )
    model.load_tokenizer(str(model_dir))
    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_pretrained(str(model_dir), local_files_only=True)
    if device_map is None:
        model = model.to(device)
    return model.eval()


def _load_qwen_pipe(model_id: str, lora_repo: str, lora_subfolder: str, dtype: torch.dtype, device: str):
    from hy_pano_qwen_pipeline import PanoDiffusionPipeline

    pipe = PanoDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True)
    pipe.load_lora_weights(
        lora_repo,
        subfolder=lora_subfolder,
        weight_name="pytorch_lora_weights.safetensors",
        torch_dtype=dtype,
    )
    return pipe.to(device)


def _generate_with_qwen(
    pipe,
    input_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
    device: str,
):
    full_positive = f"{QWEN_POSITIVE_PREFIX}{prompt}{QWEN_POSITIVE_SUFFIX}".strip()
    full_negative = f"{QWEN_NEGATIVE_PREFIX} {negative_prompt}".strip()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = dict(
        image=input_image,
        prompt=full_positive,
        negative_prompt=full_negative,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
        width=DIFFUSION_WIDTH,
        height=TARGET_HEIGHT,
    )
    result = pipe(**kwargs)
    return _finalize_panorama(result.images[0]), None


def _generate_with_hy_pano2(
    model,
    input_image_path: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
):
    global _HY_PANO_NEGATIVE_PROMPT_WARNED

    if negative_prompt.strip() and not _HY_PANO_NEGATIVE_PROMPT_WARNED:
        print("[Panorama] HY-Pano-2 full backend ignores negative_prompt; continuing with the model-native prompt flow.")
        _HY_PANO_NEGATIVE_PROMPT_WARNED = True

    model.generation_config.diff_infer_steps = int(steps)
    model.generation_config.diff_guidance_scale = float(guidance_scale)
    cot_text, outputs = model.generate_image(
        prompt=prompt,
        image=str(input_image_path),
        seed=seed,
        image_size=(TARGET_HEIGHT, DIFFUSION_WIDTH),
        bot_task="think_recaption",
        use_system_prompt="en_unified",
        max_new_tokens=2048,
        verbose=0,
    )
    output_image = outputs[0] if isinstance(outputs, list) else outputs
    recaption = cot_text[0] if isinstance(cot_text, list) and cot_text else cot_text
    return _finalize_panorama(output_image), recaption


def score_panorama(image_path: Path, moge_model: MoGeModel) -> dict:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 255.0)
    sharpness = _laplacian_var(gray)
    black_ratio = float((gray < 10).mean())
    white_ratio = float((gray > 245).mean())

    with torch.no_grad():
        pano_depth = pred_pano_depth(moge_model, image, resize_to=1024)

    pano_mask = np.array(pano_depth["mask"]).astype(bool)
    valid_ratio = float(pano_mask.mean())
    lower_valid_ratio = float(pano_mask[int(h * 0.55):].mean())
    bottom_valid_ratio = float(pano_mask[int(h * 0.75):].mean())

    depth_np = pano_depth["distance"].detach().float().cpu().numpy()
    valid_depth = depth_np[pano_mask]
    if valid_depth.size > 0:
        q10, q50, q90 = np.quantile(valid_depth, [0.1, 0.5, 0.9])
        depth_span = float((q90 - q10) / max(q50, 1e-6))
        depth_span_score = _clamp01(depth_span / 2.0)
    else:
        depth_span = 0.0
        depth_span_score = 0.0

    split_extrinsics, split_intrinsics, split_h, split_w = _build_mid_ring_cameras()
    split_images = split_panorama_image(image_np, split_extrinsics, split_intrinsics, h=split_h, w=split_w, interp=cv2.INTER_AREA)
    split_masks = split_panorama_depth(
        pano_mask.astype(np.float32),
        split_extrinsics,
        split_intrinsics,
        h=split_h,
        w=split_w,
        distance_to_depth=False,
    )

    split_valid_ratios = []
    split_sharpness = []
    split_brightness = []
    good_split_count = 0
    for split_img, split_mask in zip(split_images, split_masks):
        split_gray = cv2.cvtColor(split_img, cv2.COLOR_RGB2GRAY)
        split_valid = float((split_mask[0] > 0.5).float().mean().item())
        split_sharp = _laplacian_var(split_gray)
        split_bright = float(split_gray.mean() / 255.0)
        split_valid_ratios.append(split_valid)
        split_sharpness.append(split_sharp)
        split_brightness.append(split_bright)
        if split_valid >= 0.58 and split_sharp >= 28.0 and 0.18 <= split_bright <= 0.92:
            good_split_count += 1

    split_valid_mean = float(np.mean(split_valid_ratios))
    split_valid_min = float(np.min(split_valid_ratios))
    split_sharp_mean = float(np.mean(split_sharpness))
    split_brightness_mean = float(np.mean(split_brightness))
    good_split_ratio = float(good_split_count / max(len(split_images), 1))

    exposure_penalty = max(0.0, black_ratio - 0.03) + max(0.0, white_ratio - 0.18)
    soft_detail_penalty = (
        max(0.0, 900.0 - sharpness) / 900.0
        + max(0.0, 650.0 - split_sharp_mean) / 650.0
    )
    excessive_depth_span_penalty = max(0.0, depth_span - 4.2) / 2.0

    final_score = (
        3.2 * split_valid_mean
        + 1.8 * split_valid_min
        + 1.8 * bottom_valid_ratio
        + 1.0 * lower_valid_ratio
        + 1.0 * valid_ratio
        + 1.1 * good_split_ratio
        + 0.8 * _brightness_score(brightness)
        + 0.5 * _brightness_score(split_brightness_mean)
        + 0.6 * _contrast_score(contrast)
        + 1.4 * _sharpness_score(sharpness)
        + 1.6 * _sharpness_score(split_sharp_mean)
        + 0.7 * depth_span_score
        - 2.0 * exposure_penalty
        - 1.2 * soft_detail_penalty
        - 0.8 * excessive_depth_span_penalty
    )

    return {
        "path": str(image_path),
        "score": float(final_score),
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "black_ratio": black_ratio,
        "white_ratio": white_ratio,
        "valid_ratio": valid_ratio,
        "lower_valid_ratio": lower_valid_ratio,
        "bottom_valid_ratio": bottom_valid_ratio,
        "depth_span": depth_span,
        "split_valid_mean": split_valid_mean,
        "split_valid_min": split_valid_min,
        "split_sharp_mean": split_sharp_mean,
        "split_brightness_mean": split_brightness_mean,
        "good_split_ratio": good_split_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--backend", choices=["auto", "hy-pano2", "qwen-lora"], default=os.environ.get("HY_PANO_BACKEND", "auto"))
    parser.add_argument("--hy-pano-model", default=os.environ.get("HY_PANO_MODEL_PATH", DEFAULT_HY_PANO_MODEL))
    parser.add_argument("--model-id", default=os.environ.get("QWEN_PANO_BASE", DEFAULT_QWEN_PANO_BASE))
    parser.add_argument("--lora-repo", default=os.environ.get("HY_PANO_LORA_REPO", DEFAULT_HY_PANO_LORA_REPO))
    parser.add_argument("--lora-subfolder", default=os.environ.get("HY_PANO_LORA_SUBFOLDER", "HY-Pano-2.0"))
    parser.add_argument("--device-map", default=os.environ.get("HY_PANO_DEVICE_MAP", "auto"))
    parser.add_argument("--max-gpu-memory", default=os.environ.get("HY_PANO_MAX_GPU_MEMORY", ""))
    parser.add_argument("--max-cpu-memory", default=os.environ.get("HY_PANO_MAX_CPU_MEMORY", ""))
    parser.add_argument("--offload-dir", default=os.environ.get("HY_PANO_OFFLOAD_DIR", DEFAULT_HY_PANO_OFFLOAD_DIR))
    parser.add_argument("--scene-type", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=97)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=os.environ.get("HY_PANO_GUIDANCE_SCALE"))
    parser.add_argument("--true-cfg-scale", type=float, default=float(os.environ.get("HY_PANO_TRUE_CFG_SCALE", "7.5")))
    parser.add_argument("--moge-model", default=os.environ.get("MOGE_MODEL_ID", "/root/autodl-tmp/models/moge-2-vitl-normal/model.pt"))
    args = parser.parse_args()

    save = Path(args.save)
    run_dir = save.parent
    candidate_dir = run_dir / "panorama_candidates"
    score_path = run_dir / "panorama_candidate_scores.json"
    save.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    prompt = _prepare_prompt(args.prompt, args.scene_type)
    negative_prompt = args.negative_prompt.strip()
    pipe_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = _select_backend(args)
    # 官方 Qwen 轻量后端的室内默认调用不附加场景描述，避免长提示词引入额外建筑结构。
    if backend == "qwen-lora" and args.scene_type == "indoor" and not args.prompt.strip():
        prompt = ""
    hy_pano_device_map = _normalize_device_map(args.device_map)
    hy_pano_max_memory = _build_max_memory(args.max_gpu_memory, args.max_cpu_memory)
    guidance_scale = args.guidance_scale
    if guidance_scale is None:
        guidance_scale = 4.0 if backend == "hy-pano2" else 1.0
    print(f"[Panorama] backend={backend}")
    print(
        f"[Panorama] resolution_chain={DIFFUSION_WIDTH}x{TARGET_HEIGHT} "
        f"-> circular_blend={BLEND_WIDTH}px -> {TARGET_WIDTH}x{TARGET_HEIGHT}"
    )

    targets = []
    for index in range(args.num_candidates):
        seed = int(args.seed + index * args.seed_step)
        target = candidate_dir / f"candidate_{index:02d}_seed{seed}.png"
        targets.append({"seed": seed, "path": target})

    generated = [item for item in targets if item["path"].exists() and _is_current_panorama(item["path"])]
    for item in generated:
        print(f"[Panorama] reuse existing seed={item['seed']} -> {item['path']}")

    missing = [item for item in targets if item not in generated]
    for item in missing:
        if item["path"].exists():
            print(f"[Panorama] regenerate stale-size candidate seed={item['seed']} -> {item['path']}")
    generator_backend = None
    input_image = None
    if missing:
        if backend == "hy-pano2":
            generator_backend = _load_hy_pano_model(
                Path(args.hy_pano_model),
                pipe_dtype,
                device,
                hy_pano_device_map,
                hy_pano_max_memory,
                Path(args.offload_dir),
            )
            print(f"[Panorama] using full HY-Pano-2.0 model at {args.hy_pano_model}")
            print(f"[Panorama] device_map={hy_pano_device_map or device}; placement={_summarize_device_map(generator_backend)}")
            if hy_pano_max_memory:
                print(f"[Panorama] max_memory={hy_pano_max_memory}")
        else:
            generator_backend = _load_qwen_pipe(args.model_id, args.lora_repo, args.lora_subfolder, pipe_dtype, device)
            print(f"[Panorama] using Qwen-Image-Edit base at {args.model_id} with LoRA {args.lora_repo}/{args.lora_subfolder}")
            input_image = Image.open(args.image).convert("RGB")

    for item in missing:
        seed = item["seed"]
        target = item["path"]
        if backend == "hy-pano2":
            output_image, recaption = _generate_with_hy_pano2(
                generator_backend,
                Path(args.image),
                prompt,
                negative_prompt,
                seed,
                args.steps,
                guidance_scale,
            )
        else:
            output_image, recaption = _generate_with_qwen(
                generator_backend,
                input_image,
                prompt,
                negative_prompt,
                seed,
                args.steps,
                guidance_scale,
                args.true_cfg_scale,
                device,
            )
        output_image.save(target)
        generated.append({"seed": seed, "path": target, "recaption": recaption})
        print(f"[Panorama] generated seed={seed} -> {target}")

    if generator_backend is not None:
        del generator_backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    moge = MoGeModel.from_pretrained(args.moge_model).to(device).eval()
    scored = []
    for item in generated:
        metrics = score_panorama(item["path"], moge)
        metrics["seed"] = item["seed"]
        if item.get("recaption"):
            metrics["recaption"] = item["recaption"]
        scored.append(metrics)
        print(
            "[PanoramaScore] "
            f"seed={item['seed']} score={metrics['score']:.4f} "
            f"valid={metrics['valid_ratio']:.3f} bottom={metrics['bottom_valid_ratio']:.3f} "
            f"split={metrics['split_valid_mean']:.3f} sharp={metrics['split_sharp_mean']:.1f}"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    _force_panorama_size(Image.open(best["path"]).convert("RGB")).save(save)
    score_path.write_text(
        json.dumps(
            {
                "backend": backend,
                "true_cfg_scale": args.true_cfg_scale if backend == "qwen-lora" else None,
                "selected": best,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_candidates": args.num_candidates,
                "candidates": scored,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Panorama] selected seed={best['seed']} score={best['score']:.4f} -> {save}")
    print(save)


if __name__ == "__main__":
    main()
