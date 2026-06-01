from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
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


def _draw_text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], *, max_width: int) -> None:
    font_size = 32
    while font_size > 14:
        font = _load_font(font_size)
        widths = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            widths.append(bbox[2] - bbox[0])
        if max(widths, default=0) <= max_width - 24:
            break
        font_size -= 2
    font = _load_font(font_size)

    x, y = xy
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])

    padding = 12
    box_w = max(widths, default=0) + 2 * padding
    box_h = sum(heights) + max(0, len(lines) - 1) * 10 + 2 * padding
    draw.rounded_rectangle((x, y, x + box_w, y + box_h), radius=8, fill=(0, 0, 0, 180))

    cursor_y = y + padding
    for line, height in zip(lines, heights, strict=True):
        draw.text((x + padding, cursor_y), line, fill=(255, 255, 255), font=font)
        cursor_y += height + 10


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    delta_xy: np.ndarray,
    *,
    color: tuple[int, int, int],
    arrow_length_px: float,
    center_xy: tuple[float, float],
) -> None:
    delta_xy = np.asarray(delta_xy, dtype=np.float32)
    norm = float(np.linalg.norm(delta_xy))
    if norm <= 1e-6:
        return

    center = np.asarray(center_xy, dtype=np.float32)
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
        exp_dir = dataclasses.replace(config, exp_name=exp_name).checkpoint_dir
    else:
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
        exp_dir = experiment_dirs[0]

    step_dirs = sorted(
        (path for path in exp_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No numeric step checkpoint found under {exp_dir}")
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


def _resolve_split_episode(dataset: LeRobotDataset, split: Literal["train", "test", "all"], episode_index: int) -> tuple[int, int | None]:
    if split == "all":
        if episode_index not in dataset.meta.episodes:
            raise ValueError(f"Episode {episode_index} not found. Available range: 0..{dataset.num_episodes - 1}")
        return episode_index, None

    split_path = dataset.root / "splits.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_obj = json.loads(split_path.read_text())
    split_trials = list(split_obj[f"{split}_trials"])
    if not (0 <= episode_index < len(split_trials)):
        raise ValueError(
            f"{split} split episode_index {episode_index} is out of range. "
            f"Valid range: 0..{len(split_trials) - 1}"
        )

    all_trials = sorted([*split_obj["train_trials"], *split_obj["test_trials"]])
    trial_index = int(split_trials[episode_index])
    dataset_episode_index = all_trials.index(trial_index)
    if dataset_episode_index not in dataset.meta.episodes:
        raise ValueError(
            f"Resolved dataset episode {dataset_episode_index} for trial {trial_index}, "
            "but it does not exist in dataset metadata."
        )
    return dataset_episode_index, trial_index


def _render_frame(
    *,
    image_t: np.ndarray,
    gt_action_abs: np.ndarray,
    pred_action_abs: np.ndarray,
    state_t: np.ndarray,
    episode_index: int,
    timestep: int,
    arrow_length_px: float,
) -> np.ndarray:
    vis = Image.fromarray(image_t).convert("RGBA")
    draw = ImageDraw.Draw(vis, "RGBA")

    h, w = image_t.shape[:2]
    gt_center = (w * 0.82, h * 0.80)
    pred_center = (w * 0.82, h * 0.92)

    gt_delta_xy = gt_action_abs[:2] - state_t[:2]
    pred_delta_xy = pred_action_abs[:2] - state_t[:2]

    _draw_arrow(
        draw,
        gt_delta_xy,
        color=(0, 170, 255),
        arrow_length_px=arrow_length_px,
        center_xy=gt_center,
    )
    _draw_arrow(
        draw,
        pred_delta_xy,
        color=(255, 140, 0),
        arrow_length_px=arrow_length_px,
        center_xy=pred_center,
    )

    _draw_text_block(
        draw,
        (12, 12),
        [
            f"EP {episode_index}  T {timestep}",
            f"STATE {_format_vec(state_t)}",
            f"GT ABS   {_format_vec(gt_action_abs)}",
            f"PRED ABS {_format_vec(pred_action_abs)}",
        ],
        max_width=image_t.shape[1] - 24,
    )

    return np.asarray(vis.convert("RGB"), dtype=np.uint8)


@dataclasses.dataclass
class Args:
    config_name: str = "pi0_sensapex"
    split: Literal["train", "test", "all"] = "all"
    episode_index: int = 0
    checkpoint_dir: Path | None = None
    exp_name: str | None = None
    dataset_root: Path | None = None
    output_path: Path = Path("examples/sensapex/action_videos_absolute/episode_0000.mp4")
    default_prompt: str | None = None
    norm_stats_dir: Path | None = None
    arrow_length_px: float = 100.0
    fps: float | None = None


def main(args: Args) -> None:
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
    resolved_episode_index, resolved_trial_index = _resolve_split_episode(dataset, args.split, args.episode_index)

    policy = _policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        norm_stats=norm_stats,
    )

    ep_start = int(dataset.episode_data_index["from"][resolved_episode_index])
    ep_end = int(dataset.episode_data_index["to"][resolved_episode_index])
    num_steps = ep_end - ep_start - 1
    if num_steps <= 0:
        raise ValueError(f"Episode {resolved_episode_index} is too short for visualization.")

    video_fps = float(args.fps if args.fps is not None else dataset.meta.fps)
    output_path = args.output_path
    if output_path.name == "episode_0000.mp4":
        output_path = output_path.with_name(f"{args.split}_episode_{args.episode_index:04d}.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output_path, fps=video_fps, codec="libx264") as writer:
        for timestep in range(num_steps):
            global_index = ep_start + timestep
            sample_t = dataset[global_index]

            image_t = _to_uint8_hwc(sample_t["image"])
            state_t = _to_numpy(sample_t["state"]).astype(np.float32)

            gt_actions = _to_numpy(sample_t["actions"]).astype(np.float32)
            if gt_actions.ndim == 1:
                gt_actions = gt_actions[None, :]
            gt_action_abs = gt_actions[0]

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
            pred_action_abs = pred_actions[0]

            frame = _render_frame(
                image_t=image_t,
                gt_action_abs=gt_action_abs,
                pred_action_abs=pred_action_abs,
                state_t=state_t,
                episode_index=resolved_episode_index,
                timestep=timestep,
                arrow_length_px=args.arrow_length_px,
            )
            writer.append_data(frame)
            print(f"[{timestep + 1}/{num_steps}] appended frame")

    print(f"Saved video to: {output_path.resolve()}")
    print(f"Config: {args.config_name}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Norm stats: {norm_stats_dir}")
    print(f"Split: {args.split}")
    print(f"Split episode index: {args.episode_index}")
    print(f"Resolved dataset episode: {resolved_episode_index}")
    if resolved_trial_index is not None:
        print(f"Resolved trial index: {resolved_trial_index}")
    print(f"Frames: {num_steps}")
    print(f"FPS: {video_fps}")


if __name__ == "__main__":
    main(tyro.cli(Args))
