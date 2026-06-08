# Sensapex Robot 

## Introduction
The [Sensapex robot](https://sensapex.com/) is a micromanipulator system commonly attached to microscopes for precise micro-scale manipulation tasks.

This setup uses **two Sensapex uMps** (8 DoF total) for the microtargeting task. There is no focusing motor / ODrive in this rig.

| Joint   | Description                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------------- |
| $x_1$   | uMp 1 X position (in ${\mu}m$)                                                                       |
| $y_1$   | uMp 1 Y position (in ${\mu}m$)                                                                       |
| $z_1$   | uMp 1 Z position (in ${\mu}m$)                                                                       | 
| $d_1$   | uMp 1 diagonal direction of the pipette between X and Z (injection depth) (in ${\mu}m$)              |
| $x_2$   | uMp 2 X position (in ${\mu}m$)                                                                       |
| $y_2$   | uMp 2 Y position (in ${\mu}m$)                                                                       |
| $z_2$   | uMp 2 Z position (in ${\mu}m$)                                                                       | 
| $d_2$   | uMp 2 diagonal direction of the pipette between X and Z (injection depth) (in ${\mu}m$)              |

## Train Dataset
The training dataset for the dual-uMp microtargeting task is structured as follows:

| Field        | Description                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- |
| Image $I_t$  | Image input (downsized to 540 x 720)                                                                                   |
| Prompt $l$   | Task description (e.g. "Move the needles towards the bead")                                                            | 
| State  $q_t$ | Robot's current 8 DoF state ([$x_{1,t}$, $y_{1,t}$, $z_{1,t}$, $d_{1,t}$, $x_{2,t}$, $y_{2,t}$, $z_{2,t}$, $d_{2,t}$])  |
| Action $a_t$ | Commanded 8 DoF target ([$x_{1,t}'$, $y_{1,t}'$, $z_{1,t}'$, $d_{1,t}'$, $x_{2,t}'$, $y_{2,t}'$, $z_{2,t}'$, $d_{2,t}'$]) |

Actions are taken directly from the `target_*` columns of the teleop CSVs (i.e. the **commanded** target at each tick), not from the next-state as in the legacy 5-DoF v3 pipeline.

Most VLAs, including the ${\pi0}$ model, internally use ${\Delta}$ actions for robust cross-embodiment learning. Although the dataset stores actions in absolute form, the ${\pi0}$ training pipeline (`LeRobotSensapexDataConfig`) converts them into ${\Delta}$ actions ($a_t - q_t$) and then divides by `step_size` (default 50 µm) so the model learns roughly-integer step counts. Z-score normalization on top handles any remaining centering/scaling.

**Sampling Rate.**  
The demonstrations are recorded at an average rate of approximately 2.5 Hz.  
For compatibility with the LeRobot format, the dataset `hz` is set to 3 Hz.

**Hugging Face.** \
The dataset lives at [RaianSilex/ump_suite_robot_dataset](https://huggingface.co/datasets/RaianSilex/ump_suite_robot_dataset) (downsized images -- this is the version we train on). All six training configs in `src/openpi/training/config.py` point to this repo. If you also build a full-resolution copy with `--keep-original`, it is pushed to `RaianSilex/ump_suite_robot_dataset_raw`.

**Checkpoints.**\
You can also download the checkpoints in [choicelab/sensapex_finetuned_models](https://huggingface.co/choicelab/sensapex_finetuned_models). \
You can find LoRA fine tuned Pi0, Pi0-FAST, and Pi0.5. (Note: any legacy checkpoints there were trained on the 5-DoF v3 single-uMp dataset; you will need to re-finetune for this 8-DoF dual-uMp rig.)


## Data Conversion

`convert_data_to_lerobot.py` builds a LeRobot dataset from your raw teleop logs.

Expected layout under `--data_root`:

```
<data_root>/
    trial_1.csv
    trial_2.csv
    ...
    <image directories or files referenced by image_path>
```

Each `trial_*.csv` must include these columns:

```
current_x,  current_y,  current_z,  current_d,    # uMp 1 live
current_x2, current_y2, current_z2, current_d2,   # uMp 2 live
target_x,   target_y,   target_z,   target_d,     # uMp 1 commanded target
target_x2,  target_y2,  target_z2,  target_d2,    # uMp 2 commanded target
image_path                                        # absolute, or relative to data_root / CSV dir
```

Any extra columns (e.g. `resistance_mohm`, legacy `current_motor` / `target_motor`) are ignored.

CSVs may live directly under `--data_root` or under `--data_root/logs/` (both are auto-detected),
and `image_path` may be absolute or relative to `data_root` (or the CSV's directory).

Uninitialized commanded-target columns (any `target_*` == 0) are repaired by default
(`--fix-uninitialized-targets`): each 0 target is replaced with the matching current state, i.e.
a zero-delta "hold" at the current position. A valid Sensapex encoder position is always a large
positive count, so a 0 is a logging sentinel from before the teleop state was synced; left as-is
it would inject huge bogus deltas that corrupt the action normalization stats. The frame is kept.

Build the dataset locally (no push) to inspect it first:

```bash
uv run examples/sensapex/convert_data_to_lerobot.py --data-root dataset_csv_format --no-push-to-hub
```

Then push it (privately) to the Hub when you're happy with it:

```bash
uv run examples/sensapex/convert_data_to_lerobot.py --data-root dataset_csv_format
```

`--data-root` defaults to `dataset_csv_format`, so on a fresh session you can usually just run the
command above. The dataset is created in `~/.cache/huggingface/lerobot/RaianSilex/ump_suite_robot_dataset`
and (unless `--no-push-to-hub`) pushed to that repo.

## Fine-Tuning Sensapex 

There are 6 different variants for fine-tuning for Sensapex robot.
- Full finetuning Pi0 (name = "pi0_sensapex")
- LoRA finetuning Pi0 (name = "pi0_sensapex_low_mem_finetune")
- Full finetuning Pi0-FAST (name = "pi0_fast_sensapex")
- LoRA finetuning Pi0-FAST (name = "pi0_fast_sensapex_low_mem_finetune")
- Full finetuning Pi0.5 (name = "pi05_sensapex")
- LoRA finetuning Pi0.5 (name = "pi0.5_sensapex_low_mem_finetune")

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_sensapex
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_sensapex --exp-name=my_experiment --overwrite
```

For easy execution, here's the scripts to get normalization stats and train for each variant of Sensapex:

```bash
# Full finetuning Pi0
uv run scripts/compute_norm_stats.py --config-name pi0_sensapex
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_sensapex --exp-name=my_experiment --overwrite

# LoRA finetuning Pi0
uv run scripts/compute_norm_stats.py --config-name pi0_sensapex_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_sensapex_low_mem_finetune --exp-name=my_experiment --overwrite

# Full finetuning Pi0-FAST
uv run scripts/compute_norm_stats.py --config-name pi0_fast_sensapex
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_sensapex --exp-name=my_experiment --overwrite

# LoRA finetuning Pi0-FAST
uv run scripts/compute_norm_stats.py --config-name pi0_fast_sensapex_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_sensapex_low_mem_finetune --exp-name=my_experiment --overwrite

# Full finetuning Pi0.5
uv run scripts/compute_norm_stats.py --config-name pi05_sensapex
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_sensapex --exp-name=my_experiment --overwrite

# LoRA finetuning Pi0.5
uv run scripts/compute_norm_stats.py --config-name pi0.5_sensapex_low_mem_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0.5_sensapex_low_mem_finetune --exp-name=my_experiment --overwrite

```

## Real-time Inference

**Robot PC**

Runs:

1. ROS2 Humble
2. ump_suite package (both uMps and the camera node)
3. `examples/sensapex/main.py` from this repo — the OpenPI client. It is **not** a ros2 node and is
   **not** run with `uv run`; launch it from a ROS-sourced Python that also has `openpi-client`
   installed (see Instructions → Terminal 3).

Responsibilities

1. Reads image and 8-D robot state from ROS
2. Sends observations to the policy server
3. Executes returned action chunks on hardware

**GPU Server**

Runs:

1. OpenPI policy server
2. Finetuned checkpoint

Responsibilities

1. Receives observation via websocket
2. Runs inference
3. Returns an action chunk

**Required ROS Topics**

Subscribed by OpenPI bridge:

```bash
/camera/image/compressed        (sensor_msgs/CompressedImage)
/ump/live                       (std_msgs/Int32MultiArray)   # uMp 1: [x1,y1,z1,d1]
/ump2/live                      (std_msgs/Int32MultiArray)   # uMp 2: [x2,y2,z2,d2]
```
Published by OpenPI bridge:

```bash
/ump/target                     (std_msgs/Int32MultiArray)   # uMp 1: [x1,y1,z1,d1,speed]
/ump2/target                    (std_msgs/Int32MultiArray)   # uMp 2: [x2,y2,z2,d2,speed]
```

The action is interpreted as absolute targets:

```bash
[x1', y1', z1', d1', x2', y2', z2', d2']
```
**Policy input format**


The robot-facing module sends the following dictionary to the policy server:

```bash
{
    "observation/image": resized_image_224x224,
    "observation/state": np.array([x1, y1, z1, d1, x2, y2, z2, d2], dtype=np.float32),
    "prompt": instruction_string
}
```
The policy returns:

```bash
actions: (T, 8)
```
Each row corresponds to:

```bash
[x1_target, y1_target, z1_target, d1_target, x2_target, y2_target, z2_target, d2_target]
```

> **Safety:** Edit the per-axis `X1_MIN/X1_MAX`, ..., `D2_MIN/D2_MAX` constants and the `MAX_D*` per-tick step caps at the top of [`main.py`](main.py) to match your workspace before running on hardware. The `_clamp` helper tolerates reversed (min > max) bounds, so use whichever ordering matches your encoder convention.

## Instructions (Example)

**Terminal 1**

```bash
cd ~/ros2_ws

source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

ros2 launch ump_suite app.launch.py
```

**Terminal 2 — policy server** (GPU machine; uses the openpi `uv` env, no ROS needed)

```bash
cd ~/bsbrl-openpi
uv run scripts/serve_policy.py \
  --env SENSAPEX \
  --port 8000 \
  --default-prompt "Move the needles towards the bead" \
  policy:checkpoint \
  --policy.config pi0_sensapex_low_mem_finetune \
  --policy.dir ~/openpi_models/pi0_sensapex_lora/30000
```
Change `--policy.config` to the config you trained and `--policy.dir` to your checkpoint **step**
directory (the full step dir, which contains `params/` + `assets/` norm-stats). Wait until:

```bash
server listening on 0.0.0.0:8000
```

**Terminal 3 — the client** `examples/sensapex/main.py` (Robot PC)

`sensapex_env.py` uses `rclpy`, so the client needs **both** ROS (`rclpy`, `sensor_msgs`,
`std_msgs`) **and** `openpi_client` / `tyro` in the *same* Python. It therefore **cannot** run under
`uv run` (no `rclpy`) or in a bare ROS shell (no `openpi_client`). Install the client into your ROS
Python **once**:

```bash
source /opt/ros/humble/setup.bash
pip install --user -e ~/bsbrl-openpi/packages/openpi-client tyro pillow
# if pip refuses (externally-managed-environment): add --break-system-packages, or use an isolated venv:
#   python3 -m venv --system-site-packages ~/.venvs/sensapex_client
#   source ~/.venvs/sensapex_client/bin/activate && pip install -e ~/bsbrl-openpi/packages/openpi-client tyro pillow
```

Then run it as a **module** from the repo root (`main.py` does `from .sensapex_env import …`, so
`python3 main.py` fails on the relative import):

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
# if you used the venv: source ~/.venvs/sensapex_client/bin/activate
cd ~/bsbrl-openpi
python3 -m examples.sensapex.main \
  --remote-host 127.0.0.1 --remote-port 8000 \
  --open-loop-horizon 8 --max-timesteps 600 \
  --resize-h 224 --resize-w 224 --default-speed 100
```
The instruction is entered **at runtime** (not a flag); type it at the prompt (`q` + Enter e-stops):

```bash
[sensapex] Live preview will be saved to: sensapex_live.png
Enter instruction: Move the needles towards the bead
```

### How to check config names: ###

```bash
cd ~/bsbrl-openpi

rg -n "name=\".*sensapex" src/openpi/training/config.py
```

Currently the output may be:

```bash
987:        name="pi0_sensapex",
1013:        name="pi0_sensapex_low_mem_finetune",
1027:        name="pi0_fast_sensapex",
1047:        name="pi0_fast_sensapex_low_mem_finetune",
1066:        name="pi05_sensapex",
1086:        name="pi0.5_sensapex_low_mem_finetune",
```

### How to download a finetuned model: ###

```bash
mkdir -p ~/openpi_models
cd ~/openpi_models

hf download choicelab/sensapex_finetuned_models \
  --local-dir sensapex_finetuned_models \
  --local-dir-use-symlinks False
```
