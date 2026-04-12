# ump_suite — openpi finetuning for the dual uMp Sensapex rig

This folder documents how to finetune the π₀ / π₀-FAST / π₀.₅ models on data collected from the
**dual uMp Sensapex micromanipulator rig** (two uMp stages + one ODrive focusing motor,
controlled from ROS 2 via the `ump_suite` package) and how to serve the resulting policy.

The rig produces a **9-D** state / action vector ordered as:

```
[x1, y1, z1, d1,  x2, y2, z2, d2,  h]
 └── uMp #1 ──┘  └── uMp #2 ──┘   └─ ODrive focus motor (ticks)
```

This order must be consistent across the CSV logs, the conversion script, the policy
input/output transforms, the training config, and the robot client (`sensapex_env.py`).

It also contains a generic, step-by-step guide at the bottom for anyone who wants to adapt this
pipeline to their **own robot**.

---

## 0. One-time setup

All commands below are run from the **repo root** (`bsbrl-openpi/`) unless stated otherwise.

```bash
cd bsbrl-openpi
uv sync              # install deps (requires uv; Python 3.11)
```

GPU memory: set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before the training command to let JAX use
up to 90% of GPU memory (default is 75%).

---

## 1. Collect and lay out raw data

Each episode is one `trial_*.csv` file under `DATA_ROOT` (either directly in `DATA_ROOT/` or
under `DATA_ROOT/logs/`). The converter auto-detects both layouts. Every row is one control
tick and contains both the **current state** and the **target command** for that tick, plus a
path to the camera frame:

```
timestep,
current_x,  current_y,  current_z,  current_d,   current_motor,
target_x,   target_y,   target_z,   target_d,    target_motor,
current_x2, current_y2, current_z2, current_d2,
target_x2,  target_y2,  target_z2,  target_d2,
image_path
```

- `current_*` and `current_*2` are the live poses of uMp #1 and uMp #2 (4 axes each).
- `current_motor` / `target_motor` are the ODrive tick count for the focusing motor.
- `target_*` / `target_*2` are the absolute commands that were sent that tick — the converter
  uses these directly as the action labels (no `t+1` shifting).
- `image_path` may be absolute, or relative to `DATA_ROOT`, or relative to the CSV's folder —
  the converter tries each. Rows with an empty `image_path` are skipped.
- The converter assembles the 9-D state and action vectors in this fixed order (matching
  `sensapex_env.py`):
  ```
  [current_x, current_y, current_z, current_d,      # uMp #1
   current_x2, current_y2, current_z2, current_d2,  # uMp #2
   current_motor]                                   # ODrive h
  ```

If you ever rename the CSV headers, edit the `STATE_*_COLS` / `ACTION_*_COLS` constants at the
top of
[convert_ump_suite_robot_data_to_lerobot.py](../../examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py)
— the concatenation order there is what defines the final 9-D layout.

---

## 2. Convert raw data to a LeRobot dataset

Script: [examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py](../../examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py)

```bash
# from the repo root
uv run examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py \
    --data-root /path/to/DATA_ROOT
```

Add `--push-to-hub` if you also want to push the dataset to the Hugging Face Hub. See
[section 2a](#2a-optional-authenticate-with-hugging-face-before-using---push-to-hub) below
for the one-time auth step.

Before running, open the script and confirm the top-of-file constants match what you want:

- `REPO_NAME = "RaianSilex/ump_suite_robot_dataset"` — must match the `repo_id` used by every
  TrainConfig in [src/openpi/training/config.py](../../src/openpi/training/config.py).
- `TASK  = "Move the needle towards the bead"` — the language prompt for every sample.
- `FPS   = 3` — raise this if your logs were recorded faster and you want to keep the original
  rate.

The resulting LeRobot dataset is written to `$HF_LEROBOT_HOME/<REPO_NAME>` (default
`~/.cache/huggingface/lerobot/...`). After this step the raw CSV/PNG layout is no longer needed
for training.

### 2a. (Optional) Authenticate with Hugging Face before using `--push-to-hub`

Pushing to the Hub needs an **access token** (not your password). One-time setup:

1. Create a token at https://huggingface.co/settings/tokens with **Write** access and copy the
   `hf_...` string.
2. Log in locally — this stores the token at `~/.cache/huggingface/token`:
   ```bash
   uv run huggingface-cli login
   ```
   Paste the token when prompted. You only do this once per machine. Alternatively, export
   it as an env var for the current shell:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Make sure `REPO_NAME` in the conversion script starts with **your** username (or an org
   you belong to). Your own username always works.

Then rerun the conversion with `--push-to-hub`. The repo is created automatically on first
push, so there is no need to pre-create it in the web UI. The script pushes with
`private=True`; flip it to `False` in
[examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py](../../examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py)
if you want the dataset public.

---

## 3. Pick a training config

All ump_suite configs live in [src/openpi/training/config.py](../../src/openpi/training/config.py).
Six are pre-defined:

| Name | Model | Use when |
|---|---|---|
| `pi0_ump_suite_robot` | π₀ full finetune | Best quality, needs most VRAM |
| `pi0_ump_suite_robot_low_mem_finetune` | π₀ LoRA | π₀ on limited VRAM |
| `pi0_fast_ump_suite_robot` | π₀-FAST full | Faster, discrete action head |
| `pi0_fast_ump_suite_robot_low_mem_finetune` | π₀-FAST LoRA | π₀-FAST on limited VRAM |
| `pi05_ump_suite_robot` | π₀.₅ full | Latest model, trains on absolute actions |
| `pi05_ump_suite_robot_low_mem_finetune` | π₀.₅ LoRA | π₀.₅ on limited VRAM |

All six configs output **absolute 5-D poses** at inference, regardless of how they handle deltas
internally. π₀ and π₀-FAST convert absolute→delta for training and add the state back at
inference; π₀.₅ uses the absolute actions directly.

---

## 4. Compute normalization statistics

Run once **per config you plan to train** (the pi0/pi0-fast and pi0.5 recipes produce different
norm stats because they handle deltas differently):

```bash
# from the repo root
uv run scripts/compute_norm_stats.py --config-name pi0_ump_suite_robot
```

Stats are saved into the config's assets directory and picked up automatically at train time.
If training later errors with "missing norm stats", rerun this step for the corresponding config.

---

## 5. Train

```bash
# from the repo root
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi0_ump_suite_robot \
        --exp-name=my_first_run \
        --overwrite
```

- `pi0_ump_suite_robot` is the config name — swap it for any of the six configs above.
- `--exp-name` is your run name; checkpoints are saved to
  `checkpoints/<config_name>/<exp_name>/<step>`.
- `--overwrite` replaces any existing checkpoints for the same config + exp-name. Omit this
  flag to resume an interrupted run.
- W&B logging is on by default (set `WANDB_MODE=disabled` to turn it off).

For LoRA configs you typically do not need `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`.

---

## 6. Serve the trained policy

```bash
# from the repo root
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_ump_suite_robot \
    --policy.dir=checkpoints/pi0_ump_suite_robot/my_first_run/29999
```

This spins up a websocket policy server on port 8000. Point your ROS 2 `ump_suite` client at
`ws://<host>:8000` and send observations in the format produced by the policy's input transform:

```python
{
    "observation/image": <H, W, 3 uint8>,   # base camera frame, any resolution (resized to 224)
    "observation/state": <9,  float32>,      # current [x1, y1, z1, d1, x2, y2, z2, d2, h]
    "prompt":            "Move the needle towards the bead",
}
```

The server responds with `{"actions": <action_horizon, 9> float32}` — a chunk of absolute next
poses. Use the first one (or roll through the whole chunk at your control rate) as the command
to the dual-micromanipulator rig.

A minimal Python example of calling a policy server from your own runtime lives in
[docs/remote_inference.md](../../docs/remote_inference.md).

---

# Adapting this pipeline to your own robot

The ump_suite integration is ~150 lines spread across three files. To onboard **any robot**
(placeholder name: `my_robot`) you clone that same three-file pattern. Everywhere below, replace
`my_robot` / `MyRobot` with your own name — lowercase-with-underscores for file and config
names, CamelCase for class names.

## Step 1 — Write your data conversion script

Create `examples/my_robot/convert_my_robot_data_to_lerobot.py` (new folder + file).

Easiest path: copy
[examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py](../../examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py)
and edit:

```python
REPO_NAME = "<your_hf_username>/my_robot_dataset"   # used everywhere as repo_id
COLS      = ["joint1", "joint2", ...]               # columns in your CSV that make up state/action
TASK      = "<your language instruction>"
FPS       = <your recording rate>
```

Inside the `LeRobotDataset.create(...)` call, set `features` to your real shapes:

```python
features={
    "image":   {"dtype": "image",   "shape": (H, W, 3), "names": ["height", "width", "channel"]},
    "state":   {"dtype": "float32", "shape": (<STATE_DIM>,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (<ACTION_DIM>,), "names": ["actions"]},
    # Add "wrist_image" (or more) if your robot has extra cameras.
},
```

Then adapt the frame loop to read **your** raw data (TFDS episodes, rosbags, HDF5, whatever)
and call `ds.add_frame({...})` per timestep and `ds.save_episode()` per episode.

## Step 2 — Write your policy input/output transforms

Create `src/openpi/policies/my_robot_policy.py`. Copy
[src/openpi/policies/ump_suite_robot_policy.py](../../src/openpi/policies/ump_suite_robot_policy.py)
and change:

1. Class names: `UmpSuiteRobotInputs` → `MyRobotInputs`, `UmpSuiteRobotOutputs` → `MyRobotOutputs`.
2. **Inputs.** In `__call__`, read whatever image / state keys your dataset uses. π₀ models
   accept up to three cameras: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`. Fill the
   ones you have, zero-fill the rest, and mask missing cameras `False` for π₀ and π₀.₅, `True`
   for π₀-FAST (see the existing comment in the file).
3. **Outputs.** Change the final slice:
   ```python
   return {"actions": np.asarray(data["actions"][:, :<ACTION_DIM>])}
   ```
   This undoes the padding that `PadStatesAndActions` added at training/inference time. It
   must match your robot's real action dimension.

## Step 3 — Register a data config and training configs

Open [src/openpi/training/config.py](../../src/openpi/training/config.py) and do three things:

**a) Import your policy** near the top (next to the `ump_suite_robot_policy` import):

```python
import openpi.policies.my_robot_policy as my_robot_policy
```

**b) Add a `LeRobotMyRobotDataConfig` class.** Copy `LeRobotUmpSuiteRobotDataConfig` in the same
file and change:

- The class name to `LeRobotMyRobotDataConfig`.
- The `RepackTransform` dict keys to map **your** LeRobot dataset keys → the keys your policy
  inputs class reads (e.g. `"observation/image": "image"` means "take the `image` column from
  the dataset and expose it under the `observation/image` key"). If you have a wrist camera,
  add `"observation/wrist_image": "wrist_image"`.
- `inputs=[my_robot_policy.MyRobotInputs(...)]` and
  `outputs=[my_robot_policy.MyRobotOutputs()]`.
- `_transforms.make_bool_mask(<ACTION_DIM>)` to match your action dimension — or use the
  `(n, -m, k)` form from
  [transforms.py:make_bool_mask](../../src/openpi/transforms.py) if some dims (e.g. a gripper)
  should stay absolute while the rest become deltas.

**c) Add `TrainConfig` entries to `_CONFIGS`.** Copy the six `*_ump_suite_robot*` entries at the
bottom of the file and change:

- `name="pi0_my_robot"` (and the other five variants).
- `data=LeRobotMyRobotDataConfig(repo_id="<your_hf_username>/my_robot_dataset", ...)`.
- For π₀-FAST, set `action_dim=<ACTION_DIM>` and pick an `action_horizon` (10 is a good
  starting point for single-arm robots) and `max_token_len` (180 for single-arm, 250 for
  bimanual).
- For π₀ full and LoRA, set `action_horizon` to something compatible with your episode length
  and control rate. Every training sample needs `action_horizon` consecutive frames from the
  same episode.
- Decide `use_delta_actions`:
  - π₀ / π₀-FAST with **absolute raw actions** → `True` (converts to delta for training, back
    to absolute at inference).
  - π₀ / π₀-FAST with **already-delta raw actions** → `False`.
  - π₀.₅ → `False` (the π₀.₅ recipe trains on absolute actions directly).

## Step 4 — Run the pipeline

From the repo root:

```bash
# 1. Convert your data to LeRobot format
uv run examples/my_robot/convert_my_robot_data_to_lerobot.py --data-root /path/to/raw

# 2. Compute norm stats (once per config you plan to train)
uv run scripts/compute_norm_stats.py --config-name pi0_my_robot

# 3. Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    uv run scripts/train.py pi0_my_robot --exp-name=first_run --overwrite

# 4. Serve
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_my_robot \
    --policy.dir=checkpoints/pi0_my_robot/first_run/29999
```

## Step 5 — Wire the policy server into your robot

Your robot's control loop should, at each control step:

1. Capture the current state and camera frame(s).
2. Build a dict shaped exactly like the output of your `MyRobotInputs` transform
   (same keys, same dtypes).
3. Send it to the policy server.
4. Apply the returned `actions` chunk (usually the first action, or step through the chunk at
   your control rate).

A minimal client example is in [docs/remote_inference.md](../../docs/remote_inference.md).

---

## Checklist before kicking off a run

- [ ] `REPO_NAME` in the conversion script matches `repo_id` in every TrainConfig.
- [ ] State dim and action dim match between: raw data, conversion features, policy
      output slice, and (for π₀-FAST) `action_dim` in the model config.
- [ ] `make_bool_mask(...)` in the data config has the right length.
- [ ] Episodes are at least `action_horizon` frames long.
- [ ] Ran `compute_norm_stats.py` for the config you are about to train.
- [ ] `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` set for full finetunes on a single GPU.
