"""Convert raw 9-DoF dual-micromanipulator demos to a LeRobot dataset.

State / action layout (must match `sensapex_env.SensapexEnv.get_observation`):
    [x1, y1, z1, d1, x2, y2, z2, d2, h]
where (x1,y1,z1,d1) is the first Sensapex uMp, (x2,y2,z2,d2) is the second
Sensapex uMp, and h is the ODrive focusing motor tick count.

Expected CSV layout (one file per episode, one row per control tick):

    timestep,
    current_x,  current_y,  current_z,  current_d,   current_motor,
    target_x,   target_y,   target_z,   target_d,    target_motor,
    current_x2, current_y2, current_z2, current_d2,
    target_x2,  target_y2,  target_z2,  target_d2,
    image_path

`image_path` is either absolute or relative to `data_root`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "RaianSilex/ump_suite_robot_dataset"  # TODO: change this to the actual repo name
TASK = "Move the needle towards the bead"  # TODO: change this to the actual task name
FPS = 3  # change this if you want a different frame rate

# Column groups — edit here if you rename any CSV headers. Order inside each
# list is irrelevant *as long as* the final concatenation matches the
# `[x1, y1, z1, d1, x2, y2, z2, d2, h]` layout expected downstream.
STATE_UMP1_COLS  = ["current_x",  "current_y",  "current_z",  "current_d"]
STATE_UMP2_COLS  = ["current_x2", "current_y2", "current_z2", "current_d2"]
STATE_MOTOR_COL  = "current_motor"

ACTION_UMP1_COLS = ["target_x",  "target_y",  "target_z",  "target_d"]
ACTION_UMP2_COLS = ["target_x2", "target_y2", "target_z2", "target_d2"]
ACTION_MOTOR_COL = "target_motor"

IMAGE_PATH_COL = "image_path"


def _build_9d(df: pd.DataFrame, ump1: list[str], ump2: list[str], motor: str) -> np.ndarray:
    """Stack the 4+4+1 columns into a (T, 9) float32 array."""
    return np.concatenate(
        [
            df[ump1].to_numpy(dtype=np.float32),
            df[ump2].to_numpy(dtype=np.float32),
            df[[motor]].to_numpy(dtype=np.float32),
        ],
        axis=-1,
    )


def _resolve_image_path(raw: str, data_root: Path, csv_path: Path) -> Path:
    p = Path(str(raw))
    if p.is_absolute() and p.exists():
        return p
    # Try data_root first, then the CSV's own directory as a fallback.
    for base in (data_root, csv_path.parent):
        candidate = base / p
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate image referenced in {csv_path}: {raw!r} "
        f"(tried absolute, {data_root}/{raw}, {csv_path.parent}/{raw})"
    )


def main(data_root: Path, *, push_to_hub: bool = False):
    out = HF_LEROBOT_HOME / REPO_NAME
    if out.exists():
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ump_suite_robot",
        fps=FPS,
        features={
            "image":   {"dtype": "image",   "shape": (224, 224, 3),
                        "names": ["height", "width", "channel"]},
            "state":   {"dtype": "float32", "shape": (9,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (9,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Accept both `data_root/*.csv` and `data_root/logs/*.csv`.
    csv_files = sorted(data_root.glob("trial_*.csv"))
    if not csv_files:
        csv_files = sorted((data_root / "logs").glob("trial_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No trial_*.csv files found under {data_root} or {data_root}/logs"
        )

    for csv_path in csv_files:
        trial = csv_path.stem  # e.g. "trial_01"
        df = pd.read_csv(csv_path)

        # Drop rows with no image path recorded (e.g. pre-recording padding).
        df = df[df[IMAGE_PATH_COL].astype(str).str.len() > 0].reset_index(drop=True)
        if len(df) == 0:
            print(f"skipping {trial}: no valid rows")
            continue

        states  = _build_9d(df, STATE_UMP1_COLS,  STATE_UMP2_COLS,  STATE_MOTOR_COL)
        actions = _build_9d(df, ACTION_UMP1_COLS, ACTION_UMP2_COLS, ACTION_MOTOR_COL)

        written = 0
        for t in range(len(df)):
            ipath = _resolve_image_path(df[IMAGE_PATH_COL].iloc[t], data_root, csv_path)
            img = np.asarray(
                Image.open(ipath).convert("RGB").resize((224, 224)),
                dtype=np.uint8,
            )
            ds.add_frame({
                "image":   img,
                "state":   states[t],
                "actions": actions[t],
                "task":    TASK,
            })
            written += 1
        ds.save_episode()
        print(f"wrote {written} frames for {trial}")

    if push_to_hub:
        ds.push_to_hub(private=True, push_videos=True, license="apache-2.0")


if __name__ == "__main__":
    tyro.cli(main)
