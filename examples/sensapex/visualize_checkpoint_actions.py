from __future__ import annotations

import dataclasses
from pathlib import Path
import random

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def _to_uint8_hwc(image) -> np.ndarray:
    image = _to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.max(image)) if image.size > 0 else 0.0
        if max_val <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _format_vec(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec), precision=3, floatmode="fixed", suppress_small=False)


def _draw_text_block(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    *,
    max_width: int,
) -> None:
    font_size = 32
    font = _load_font(font_size)
    while font_size > 14:
        font = _load_font(font_size)
        widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(bbox[2] - bbox[0])
        if max(widths, default=0) <= max_width - 24:
            break
        font_size -= 2

    x, y = xy
    line_heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])

    padding = 12
    box_w = max(widths, default=0) + 2 * padding
    box_h = sum(line_heights) + max(0, len(lines) - 1) * 10 + 2 * padding
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=8, fill=(0, 0, 0, 180))

    cursor_y = y + padding
    for line, line_h in zip(lines, line_heights, strict=True):
        draw.text((x + padding, cursor_y), line, fill=(255, 255, 255), font=font)
        cursor_y += line_h + 10


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    image_hw: tuple[int, int],
    delta_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    arrow_length_px: float,
) -> None:
    h, w = image_hw
    center = np.array([w * 0.84, h * 0.82], dtype=np.float32)

    delta_xy = np.asarray(delta_xy, dtype=np.float32)
    norm = float(np.linalg.norm(delta_xy))
    if norm <= 1e-6:
        return

    vector = delta_xy / norm * float(arrow_length_px)

    end = center + vector
    draw.line((center[0], center[1], end[0], end[1]), fill=color, width=9)

    direction = vector / np.linalg.norm(vector)
    perpendicular = np.array([-direction[1], direction[0]], dtype=np.float32)
    head_len = min(36.0, max(18.0, 0.3 * np.linalg.norm(vector)))
    left = end - head_len * direction + 0.5 * head_len * perpendicular
    right = end - head_len * direction - 0.5 * head_len * perpendicular
    draw.polygon([tuple(end), tuple(left), tuple(right)], fill=color)


def _resolve_norm_stats_dir(config: _config.TrainConfig, asset_id: str | None, checkpoint_dir: Path) -> Path:
    candidates: list[Path] = []
    if asset_id is not None:
        candidates.append(checkpoint_dir / "assets" / asset_id)
        candidates.append(config.assets_dirs / asset_id)
        candidates.extend(Path("assets").glob(f"*/{asset_id}"))

    for candidate in candidates:
        if (candidate / "norm_stats.json").is_file():
            return candidate

    searched = "\n".join(str(path / "norm_stats.json") for path in candidates)
    raise FileNotFoundError(f"Could not find norm stats for asset_id={asset_id!r}. Searched:\n{searched}")


def _resolve_checkpoint_dir(config: _config.TrainConfig, exp_name: str | None, checkpoint_dir: Path | None) -> Path:
    if checkpoint_dir is not None:
        return checkpoint_dir.resolve()

    if exp_name is not None:
        return dataclasses.replace(config, exp_name=exp_name).checkpoint_dir

    config_root = Path(config.checkpoint_base_dir) / config.name
    if not config_root.is_dir():
        raise FileNotFoundError(
            f"Checkpoint root not found: {config_root}. Pass --checkpoint-dir or --exp-name explicitly."
        )

    experiment_dirs = sorted(path for path in config_root.iterdir() if path.is_dir())
    if len(experiment_dirs) != 1:
        raise ValueError(
            f"Expected exactly one experiment dir under {config_root}, found {len(experiment_dirs)}. "
            "Pass --exp-name or --checkpoint-dir explicitly."
        )

    step_dirs = sorted(
        (path for path in experiment_dirs[0].iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No numeric step checkpoint found under {experiment_dirs[0]}")
    return step_dirs[-1].resolve()


def _make_dataset(
    repo_id: str,
    root: Path | None,
    action_horizon: int,
    action_sequence_keys: tuple[str, ...] | list[str],
) -> LeRobotDataset:
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {key: [t / meta.fps for t in range(action_horizon)] for key in action_sequence_keys}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)


def _sample_episode_and_timestep(dataset: LeRobotDataset, rng: random.Random) -> tuple[int, int, int]:
    valid_episode_indices = [
        ep_idx for ep_idx, ep in dataset.meta.episodes.items() if int(ep["length"]) >= 2
    ]
    if not valid_episode_indices:
        raise ValueError("Dataset must contain at least one episode with length >= 2.")

    episode_index = rng.choice(valid_episode_indices)
    episode_start = int(dataset.episode_data_index["from"][episode_index])
    episode_end = int(dataset.episode_data_index["to"][episode_index])
    timestep = rng.randrange(0, episode_end - episode_start - 1)
    global_index = episode_start + timestep
    return episode_index, timestep, global_index


@dataclasses.dataclass
class Args:
    config_name: str = "pi0_sensapex"
    checkpoint_dir: Path | None = None
    exp_name: str | None = None
    dataset_root: Path | None = None
    output_dir: Path = Path("examples/sensapex/action_visualizations")
    num_samples: int = 20
    seed: int = 0
    default_prompt: str | None = None
    norm_stats_dir: Path | None = None
    arrow_length_px: float = 220.0


def _render_visualization(
    *,
    image_t: np.ndarray,
    image_tp1: np.ndarray,
    state_t: np.ndarray,
    gt_action: np.ndarray,
    pred_action: np.ndarray,
    prompt: str,
    checkpoint_dir: Path,
    config_name: str,
    episode_index: int,
    timestep: int,
    arrow_length_px: float,
) -> Image.Image:
    vis_left = Image.fromarray(image_t).convert("RGBA")
    draw = ImageDraw.Draw(vis_left, "RGBA")

    _draw_arrow(
        draw,
        image_t.shape[:2],
        gt_action[:2],
        color=(0, 170, 255),
        arrow_length_px=arrow_length_px,
    )
    _draw_arrow(
        draw,
        image_t.shape[:2],
        pred_action[:2],
        color=(255, 140, 0),
        arrow_length_px=arrow_length_px,
    )

    _draw_text_block(
        draw,
        (12, 12),
        [
            f"GT  {_format_vec(gt_action)}",
            f"PRED {_format_vec(pred_action)}",
        ],
        max_width=image_t.shape[1] - 24,
    )

    left = vis_left.convert("RGB")
    right = Image.fromarray(image_tp1).convert("RGB")
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), (0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def main(args: Args) -> None:
    rng = random.Random(args.seed)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be >= 1")

    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError(f"Config {args.config_name!r} does not define a dataset repo_id.")

    checkpoint_dir = _resolve_checkpoint_dir(config, args.exp_name, args.checkpoint_dir)
    norm_stats_dir = (
        args.norm_stats_dir.resolve()
        if args.norm_stats_dir is not None
        else _resolve_norm_stats_dir(config, data_config.asset_id, checkpoint_dir)
    )
    norm_stats = _normalize.load(norm_stats_dir)

    dataset = _make_dataset(
        data_config.repo_id,
        args.dataset_root.resolve() if args.dataset_root is not None else None,
        config.model.action_horizon,
        tuple(data_config.action_sequence_keys),
    )

    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_available = sum(max(0, int(ep["length"]) - 1) for ep in dataset.meta.episodes.values())
    target_samples = min(args.num_samples, max_available)
    if target_samples < args.num_samples:
        print(f"Requested {args.num_samples} samples, but only {max_available} valid timesteps are available.")

    seen_indices: set[int] = set()
    sample_count = 0
    while sample_count < target_samples:
        episode_index, timestep, global_index = _sample_episode_and_timestep(dataset, rng)
        if global_index in seen_indices:
            continue
        seen_indices.add(global_index)

        sample_t = dataset[global_index]
        sample_tp1 = dataset[global_index + 1]

        image_t = _to_uint8_hwc(sample_t["image"])
        image_tp1 = _to_uint8_hwc(sample_tp1["image"])
        state_t = _to_numpy(sample_t["state"]).astype(np.float32)
        gt_actions = _to_numpy(sample_t["actions"]).astype(np.float32)
        if gt_actions.ndim == 1:
            gt_actions = gt_actions[None, :]

        prompt = args.default_prompt or sample_t.get("task")
        if prompt is None:
            raise ValueError("No prompt found in dataset sample and --default-prompt was not provided.")

        pred = policy.infer(
            {
                "observation/image": image_t,
                "observation/state": state_t,
                "prompt": prompt,
            }
        )
        pred_actions = _to_numpy(pred["actions"]).astype(np.float32)
        if pred_actions.ndim == 1:
            pred_actions = pred_actions[None, :]

        gt_action = gt_actions[0]
        pred_action = pred_actions[0]
        canvas = _render_visualization(
            image_t=image_t,
            image_tp1=image_tp1,
            state_t=state_t,
            gt_action=gt_action,
            pred_action=pred_action,
            prompt=prompt,
            checkpoint_dir=checkpoint_dir,
            config_name=args.config_name,
            episode_index=episode_index,
            timestep=timestep,
            arrow_length_px=args.arrow_length_px,
        )

        output_path = args.output_dir / (
            f"sample_{sample_count:03d}_ep_{episode_index:04d}_t_{timestep:04d}.png"
        )
        canvas.save(output_path)

        print(f"[{sample_count + 1}/{target_samples}] saved: {output_path.resolve()}")
        print(f"  episode={episode_index}, timestep={timestep}, global_index={global_index}")
        print(f"  gt_action[0]={gt_action}")
        print(f"  pred_action[0]={pred_action}")
        sample_count += 1

    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Norm stats: {norm_stats_dir}")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main(tyro.cli(Args))
