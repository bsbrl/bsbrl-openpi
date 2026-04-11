"""Convert raw 5-DoF micromanipulator demos to a LeRobot dataset."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "RaianSilex/ump_suite_robot_dataset"  # TODO: change this to the actual repo name
COLS = ["x", "y", "z", "d", "h"]
TASK = "Move the needle towards the bead"  # TODO: change this to the actual task name
FPS = 3  # change this if you want a different frame rate

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
            "state":   {"dtype": "float32", "shape": (5,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (5,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for csv in sorted((data_root / "logs").glob("trial_*.csv")):
        trial = csv.stem  # e.g. "trial_01"
        df = pd.read_csv(csv)[COLS].to_numpy(dtype=np.float32)
        frames = sorted((data_root / "frames" / trial).glob("frame_*.png"))

        T = min(len(df), len(frames)) - 1  # -1 since action_t = state_{t+1}
        for t in range(T):
            img = np.asarray(Image.open(frames[t]).convert("RGB").resize((224, 224)),
                             dtype=np.uint8)
            ds.add_frame({
                "image":   img,
                "state":   df[t],
                "actions": df[t + 1],  # next absolute pose — the command
                "task":    TASK,
            })
        ds.save_episode()
        print(f"wrote {T} frames for {trial}")

    if push_to_hub:
        ds.push_to_hub(private=True, push_videos=True, license="apache-2.0")

if __name__ == "__main__":
    tyro.cli(main)