"""Create LeRobot datasets for the dual-uMp Sensapex rig (8-DOF, no motor).

State / action layout (must match `sensapex_env.SensapexEnv.get_observation`):
    [x1, y1, z1, d1, x2, y2, z2, d2]

Each per-trial CSV is expected to contain at minimum these columns:
    current_x,  current_y,  current_z,  current_d,    (uMp 1 live)
    current_x2, current_y2, current_z2, current_d2,   (uMp 2 live)
    target_x,   target_y,   target_z,   target_d,     (uMp 1 commanded target)
    target_x2,  target_y2,  target_z2,  target_d2,    (uMp 2 commanded target)
    image_path                                        (absolute, or relative
                                                      to data_root / CSV dir)

Any other columns (e.g. `resistance_mohm`, legacy `current_motor` / `target_motor`)
are ignored. Actions are taken straight from the `target_*` columns -- this differs
from the old single-uMp v3 pipeline which used next-state as the action. Downstream
training (`LeRobotSensapexDataConfig`) still converts these absolute targets
into deltas + step-scaled units before normalization.

Uninitialized commanded-target columns (any `target_*` == 0) are repaired by default
(see ``fix_uninitialized_targets``): a valid Sensapex encoder position is always a
large positive count, so a 0 is a logging sentinel from before the teleop state was
synced. Each 0 target is replaced with the matching current state -- i.e. a zero-delta
"hold" at the current position -- instead of being left as an enormous bogus jump that
would wreck the action normalization stats. The frame (and its image) is kept.

Outputs (one or two LeRobot repos under HF_LEROBOT_HOME):
  HF_LEROBOT_HOME/<down_repo_id>   (downsized images -- the one we train on)
  HF_LEROBOT_HOME/<raw_repo_id>    (full-res images, only when --keep-original)
Both default to the RaianSilex/ump_suite_robot_dataset namespace.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from PIL import Image
import tyro

STATE_COLS = [
    "current_x", "current_y", "current_z", "current_d",
    "current_x2", "current_y2", "current_z2", "current_d2",
]
ACTION_COLS = [
    "target_x", "target_y", "target_z", "target_d",
    "target_x2", "target_y2", "target_z2", "target_d2",
]
IMAGE_PATH_COL = "image_path"


def _trial_idx_from_name(p: Path) -> int:
    m = re.search(r"trial_(\d+)", p.stem)
    if not m:
        raise ValueError(f"Cannot parse trial index from: {p.name}")
    return int(m.group(1))


def _load_rgb_uint8(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _resize_rgb_uint8(img_arr: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """
    Resize to exactly (out_h, out_w). Aspect ratio is NOT preserved here.
    Use only if you are sure you want exact resize.
    """
    out_h, out_w = out_hw
    im = Image.fromarray(img_arr)
    im = im.resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _resize_keep_aspect_to_fit(img_arr: np.ndarray, out_hw: tuple[int, int], pad_value: int = 0) -> np.ndarray:
    """
    Resize with aspect ratio preserved and pad to the target size (letterbox).
    out_hw is (H, W).
    """
    out_h, out_w = out_hw
    im = Image.fromarray(img_arr)
    in_w, in_h = im.size

    scale = min(out_w / in_w, out_h / in_h)
    new_w = int(round(in_w * scale))
    new_h = int(round(in_h * scale))
    im_rs = im.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (out_w, out_h), (pad_value, pad_value, pad_value))
    left = (out_w - new_w) // 2
    top = (out_h - new_h) // 2
    canvas.paste(im_rs, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def _resolve_image_path(raw: str, data_root: Path, csv_path: Path) -> Path:
    p = Path(str(raw))
    if p.is_absolute() and p.exists():
        return p
    for base in (data_root, csv_path.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate image referenced in {csv_path}: {raw!r} "
        f"(tried absolute, {data_root}/{raw}, {csv_path.parent}/{raw})"
    )


def _build_repo_suffix(stop_padding: int) -> str:
    if stop_padding > 0:
        return f"stop{stop_padding}"
    return ""


def _build_8d(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=np.float32)


@dataclass
class Args:
    data_root: Path = Path("dataset_csv_format")

    keep_original: bool = False

    # Repair uninitialized commanded-target columns (any target_* == 0) by replacing each 0 with
    # the matching current state, i.e. a zero-delta "hold" at the current position. Valid Sensapex
    # encoder positions are always large positive counts, so a 0 is a logging sentinel from before
    # the teleop state was synced (observed in trials 39 and 59). This keeps the frame and its image
    # instead of dropping it.
    fix_uninitialized_targets: bool = True

    # Number of "stay still" frames to append at the end of each episode.
    # 0 = vanilla (no padding). When > 0, the last observation is repeated N times
    # with actions set to the final state (absolute target = current position).
    stop_padding: int = 0

    # Output repos (built dynamically in __post_init__)
    raw_repo_id: str = ""
    down_repo_id: str = ""

    def __post_init__(self):
        suffix = _build_repo_suffix(self.stop_padding)
        tag = f"_{suffix}" if suffix else ""
        if not self.raw_repo_id:
            self.raw_repo_id = f"RaianSilex/ump_suite_robot_dataset_raw{tag}"
        if not self.down_repo_id:
            self.down_repo_id = f"RaianSilex/ump_suite_robot_dataset{tag}"

    # Dataset metadata
    robot_type: str = "ump_suite_robot"
    fps: int = 3
    task_text: str = "Move the needles towards the bead"

    # Downscale target (H, W). For 1440x1080 (4:3), recommend 540x720.
    down_h: int = 540
    down_w: int = 720

    # If True, downscaled images are aspect-preserving + padded (letterbox) to (down_h, down_w).
    # If False, images are resized exactly to (down_h, down_w) (may distort aspect ratio).
    down_keep_aspect_and_pad: bool = True

    # Writer parallelism
    image_writer_threads: int = 10
    image_writer_processes: int = 5

    # Overwrite existing output repos in HF_LEROBOT_HOME
    overwrite: bool = True

    test_frac: float = 0.10
    seed: int = 0
    test_trials: list[int] | None = None  # if provided, overrides random split

    push_to_hub: bool = True


def _prepare_out_dir(repo_id: str, overwrite: bool) -> Path:
    out_path = HF_LEROBOT_HOME / repo_id
    if out_path.exists():
        if overwrite:
            shutil.rmtree(out_path)
        else:
            raise FileExistsError(f"Output exists: {out_path} (set overwrite=True)")
    return out_path


def _create_dataset(repo_id: str, args: Args, image_shape: tuple[int, int, int]) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,  # (H, W, 3)
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
        },
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def main(args: Args) -> None:
    data_root = args.data_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {data_root}")

    # Accept both `data_root/*.csv` and `data_root/logs/*.csv`.
    csv_files = sorted(data_root.glob("trial_*.csv"))
    if not csv_files:
        csv_files = sorted((data_root / "logs").glob("trial_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No trial_*.csv files found under {data_root} or {data_root}/logs")

    csv_files = sorted(csv_files, key=_trial_idx_from_name)
    trial_indices = sorted({_trial_idx_from_name(p) for p in csv_files})

    if args.test_trials is not None and len(args.test_trials) > 0:
        test_set = set(args.test_trials)
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(trial_indices)
        k = max(1, int(round(len(trial_indices) * args.test_frac)))
        test_set = {int(v) for v in perm[:k]}

    train_set = [t for t in trial_indices if t not in test_set]
    test_list = [t for t in trial_indices if t in test_set]

    print(f"Split: train={len(train_set)}, test={len(test_list)}")
    print(f"Test trials: {test_list}")

    # Peek at the first usable frame to determine raw image shape (for dataset schema).
    first_csv = csv_files[0]
    first_df = pd.read_csv(first_csv)
    first_image_path = _resolve_image_path(
        str(first_df[IMAGE_PATH_COL].iloc[0]), data_root, first_csv
    )
    raw0 = _load_rgb_uint8(first_image_path)
    raw_h, raw_w = raw0.shape[0], raw0.shape[1]

    # Prepare output dirs (delete if overwrite)
    if args.keep_original:
        _prepare_out_dir(args.raw_repo_id, args.overwrite)
    _prepare_out_dir(args.down_repo_id, args.overwrite)

    # Create datasets
    if args.keep_original:
        raw_ds = _create_dataset(args.raw_repo_id, args, image_shape=(raw_h, raw_w, 3))
    down_ds = _create_dataset(args.down_repo_id, args, image_shape=(args.down_h, args.down_w, 3))

    for csv_path in csv_files:
        trial_idx = _trial_idx_from_name(csv_path)
        df = pd.read_csv(csv_path)

        missing = [c for c in (*STATE_COLS, *ACTION_COLS, IMAGE_PATH_COL) if c not in df.columns]
        if missing:
            raise KeyError(f"{csv_path.name} missing columns: {missing}")

        # Drop rows without an image (e.g. pre-recording padding).
        # NaNs stringify to "nan", so we need an explicit null check here.
        image_paths = df[IMAGE_PATH_COL]
        df = df[image_paths.notna() & image_paths.astype(str).str.strip().ne("")].reset_index(drop=True)
        num_rows = len(df)
        if num_rows == 0:
            print(f"[skip] trial_{trial_idx}: no valid rows")
            continue

        # Repair uninitialized commanded-target columns (target_* == 0). A valid Sensapex encoder
        # position is always a large positive count, so a 0 is a logging sentinel from before the
        # teleop state was synced -- the intended command at that instant is simply to hold the
        # current position. Replace each 0 target with the matching current state (per column), so
        # the action becomes a zero-delta "hold" instead of an enormous bogus jump. The frame (and
        # its image) is kept. STATE_COLS and ACTION_COLS are positionally aligned per DoF.
        if args.fix_uninitialized_targets:
            state_vals = df[STATE_COLS].to_numpy(dtype=np.float32)
            target_vals = df[ACTION_COLS].to_numpy(dtype=np.float32)
            mask0 = target_vals == 0
            n_fixed = int(mask0.any(axis=1).sum())
            if n_fixed:
                df[ACTION_COLS] = np.where(mask0, state_vals, target_vals)
                print(f"[clean] trial_{trial_idx}: held {n_fixed} uninitialized-target row(s) at current state")

        states = _build_8d(df, STATE_COLS)
        actions = _build_8d(df, ACTION_COLS)

        for t in range(num_rows):
            ipath = _resolve_image_path(str(df[IMAGE_PATH_COL].iloc[t]), data_root, csv_path)
            img_raw = _load_rgb_uint8(ipath)

            if args.keep_original:
                raw_ds.add_frame({
                    "image": img_raw,
                    "state": states[t],
                    "actions": actions[t],
                    "task": args.task_text,
                })

            if args.down_keep_aspect_and_pad:
                img_down = _resize_keep_aspect_to_fit(img_raw, (args.down_h, args.down_w))
            else:
                img_down = _resize_rgb_uint8(img_raw, (args.down_h, args.down_w))

            down_ds.add_frame({
                "image": img_down,
                "state": states[t],
                "actions": actions[t],
                "task": args.task_text,
            })

        # Append "stay still" frames using the true terminal observation.
        # The last recorded state is repeated `stop_padding` times with
        # action = state (absolute target = current position, i.e. no movement).
        if args.stop_padding > 0:
            terminal_state = states[-1]
            stop_action = terminal_state.copy()

            last_image_path = _resolve_image_path(
                str(df[IMAGE_PATH_COL].iloc[-1]), data_root, csv_path
            )
            last_img_raw = _load_rgb_uint8(last_image_path)
            if args.down_keep_aspect_and_pad:
                last_img_down = _resize_keep_aspect_to_fit(last_img_raw, (args.down_h, args.down_w))
            else:
                last_img_down = _resize_rgb_uint8(last_img_raw, (args.down_h, args.down_w))

            for _ in range(args.stop_padding):
                if args.keep_original:
                    raw_ds.add_frame({
                        "image": last_img_raw,
                        "state": terminal_state,
                        "actions": stop_action,
                        "task": args.task_text,
                    })
                down_ds.add_frame({
                    "image": last_img_down,
                    "state": terminal_state,
                    "actions": stop_action,
                    "task": args.task_text,
                })

        if args.keep_original:
            raw_ds.save_episode()
        down_ds.save_episode()
        pad_info = f" + {args.stop_padding} stop frames" if args.stop_padding > 0 else ""
        print(f"[OK] trial_{trial_idx}: wrote {num_rows}{pad_info} frames")

    split_obj = {
        "seed": args.seed,
        "test_frac": args.test_frac,
        "train_trials": train_set,
        "test_trials": test_list,
    }
    if args.keep_original:
        raw_root = HF_LEROBOT_HOME / args.raw_repo_id
        raw_root.mkdir(parents=True, exist_ok=True)
        with open(raw_root / "splits.json", "w") as f:
            json.dump(split_obj, f, indent=2)
        print(f"Raw dataset saved to:  {HF_LEROBOT_HOME / args.raw_repo_id}")

    down_root = HF_LEROBOT_HOME / args.down_repo_id
    down_root.mkdir(parents=True, exist_ok=True)
    with open(down_root / "splits.json", "w") as f:
        json.dump(split_obj, f, indent=2)

    print("Done.")
    print(f"Down dataset saved to: {HF_LEROBOT_HOME / args.down_repo_id}")

    if args.push_to_hub:
        tags = ["sensapex", "microtargeting", "micromanipulation", "dual_ump"]
        if args.keep_original:
            raw_ds.push_to_hub(
                tags=tags,
                private=True,
                push_videos=True,
                license="apache-2.0",
            )
        down_ds.push_to_hub(
            tags=tags,
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
