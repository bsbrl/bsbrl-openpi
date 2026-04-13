# ump_suite in openpi

This folder documents how to use `openpi` with the dual Sensapex uMp micromanipulator setup driven by
`ump_suite`, and how to adapt the same pattern to a different robot.

The `ump_suite_robot` integration currently consists of three pieces:

- [examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py](../../examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py)
- [src/openpi/policies/ump_suite_robot_policy.py](../../src/openpi/policies/ump_suite_robot_policy.py)
- [src/openpi/training/config.py](../../src/openpi/training/config.py)

At a high level, the pipeline is:

1. Record demonstrations from the robot.
2. Convert them to a LeRobot dataset.
3. Compute normalization statistics.
4. Fine-tune a VLA checkpoint.
5. Serve the trained checkpoint with `serve_policy.py`.
6. Send live observations from the robot runtime and execute returned actions.

## Robot layout

The rig uses a 9-D state/action vector:

```text
[x1, y1, z1, d1,  x2, y2, z2, d2,  h]
 └── uMp #1 ──┘  └── uMp #2 ──┘   └─ ODrive focus motor
```

This ordering must stay consistent across:

- raw demonstrations
- the LeRobot conversion script
- the policy input/output transforms
- the training config
- the robot runtime that talks to the policy server

## 1. Raw demonstration format

Each episode is one `trial_*.csv` file. The converter accepts either:

- `DATA_ROOT/trial_*.csv`
- `DATA_ROOT/logs/trial_*.csv`

Each CSV row is one control tick and should contain:

```text
timestep,
current_x,  current_y,  current_z,  current_d,   current_motor,
target_x,   target_y,   target_z,   target_d,    target_motor,
current_x2, current_y2, current_z2, current_d2,
target_x2,  target_y2,  target_z2,  target_d2,
image_path
```

Semantics:

- `current_*` and `current_*2` are the observed robot state at that tick.
- `target_*` and `target_*2` are the commands issued at that same tick.
- `current_motor` and `target_motor` are the focus motor values.
- `image_path` can be absolute, relative to `DATA_ROOT`, or relative to the CSV directory.

The converter builds:

```text
state   = [current_x, current_y, current_z, current_d, current_x2, current_y2, current_z2, current_d2, current_motor]
actions = [target_x,  target_y,  target_z,  target_d,  target_x2,  target_y2,  target_z2,  target_d2,  target_motor]
```

Important detail: this integration uses the command logged on the same row as the supervision target. It does not
shift actions by `t+1`.

If you rename CSV headers, update the column constants at the top of
[convert_ump_suite_robot_data_to_lerobot.py](../../examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py).

## 2. Convert demonstrations to LeRobot

From the repo root:

```bash
uv sync
uv run examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py \
    --data-root /path/to/DATA_ROOT
```

Before you run it, confirm these constants in the converter:

- `REPO_NAME`: the LeRobot dataset repo id, for example `your_hf_username/ump_suite_robot_dataset`
- `TASK`: the language instruction stored in every frame
- `FPS`: the control/recording rate you want reflected in the dataset metadata

The converter creates a LeRobot dataset with:

- `image`: one RGB frame per step
- `state`: shape `(9,)`
- `actions`: shape `(9,)`
- `task`: the prompt used later during training via `prompt_from_task=True`

The output is written to `$HF_LEROBOT_HOME/<REPO_NAME>`.

### Optional: push the dataset to Hugging Face Hub

If you want the converter to upload the dataset:

```bash
uv run huggingface-cli login
uv run examples/ump_suite_robot/convert_ump_suite_robot_data_to_lerobot.py \
    --data-root /path/to/DATA_ROOT \
    --push-to-hub
```

Make sure `REPO_NAME` starts with a namespace you control.

## 3. Training configs

`ump_suite_robot` is already registered in
[src/openpi/training/config.py](../../src/openpi/training/config.py).

Available configs:

| Config | Model | Notes |
|---|---|---|
| `pi0_ump_suite_robot` | pi0 full finetune | Highest memory use |
| `pi0_ump_suite_robot_low_mem_finetune` | pi0 LoRA | Lower-memory pi0 |
| `pi0_fast_ump_suite_robot` | pi0-FAST full finetune | Faster autoregressive policy |
| `pi0_fast_ump_suite_robot_low_mem_finetune` | pi0-FAST LoRA | Lower-memory pi0-FAST |
| `pi05_ump_suite_robot` | pi0.5 full finetune | Absolute-action recipe |
| `pi05_ump_suite_robot_low_mem_finetune` | pi0.5 LoRA | Lower-memory pi0.5 |

Action handling:

- pi0 and pi0-FAST train on deltas here: the config subtracts state from the logged absolute action during training and
  adds it back at inference.
- pi0.5 trains directly on absolute actions.
- All six configs return absolute 9-D actions at inference.

One thing to double-check before training: the `repo_id` in the config entries must match the `REPO_NAME` used by the
converter.

## 4. Compute normalization statistics

Run this once per config you plan to train:

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_ump_suite_robot
```

Do this separately for pi0/pi0-FAST vs pi0.5 if you plan to train both, because the action preprocessing differs.

## 5. Train a checkpoint

Example:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_ump_suite_robot \
    --exp-name=my_first_run \
    --overwrite
```

Notes:

- Checkpoints are written to `checkpoints/<config_name>/<exp_name>/<step>`.
- Omit `--overwrite` if you want to preserve an existing run directory.
- Set `WANDB_MODE=disabled` if you do not want Weights & Biases logging.
- The full finetune configs are much heavier than the LoRA variants.

## 6. Serve the trained model

Example:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_ump_suite_robot \
    --policy.dir=checkpoints/pi0_ump_suite_robot/my_first_run/29999
```

This starts the websocket policy server on port `8000`.

Your robot runtime should send observations shaped like:

```python
{
    "observation/image": <H, W, 3 uint8>,
    "observation/state": <9 float32>,
    "prompt": "Move the needle towards the bead",
}
```

The response is:

```python
{"actions": <action_horizon, 9 float32>}
```

Those are absolute actions in the same 9-D layout as the demonstrations. In most runtimes you either:

- execute only the first action, then re-query the policy
- execute a short prefix of the chunk at your control rate, then re-query

For a minimal client example, see [docs/remote_inference.md](../../docs/remote_inference.md).

## End-to-end summary for this robot

From demonstrations to live inference, the data path is:

1. The robot logger writes `trial_*.csv` files plus image paths.
2. The converter reads those files and creates a LeRobot dataset with `image`, `state`, `actions`, and `task`.
3. `prompt_from_task=True` turns the LeRobot `task` into the model `prompt` during training.
4. The ump suite data config repacks dataset fields into inference-style keys.
5. The policy transform converts that into the model input schema and slices the model output back to 9 dimensions.
6. `serve_policy.py` loads the trained checkpoint and exposes it over websocket.
7. The robot runtime sends live `observation/image`, `observation/state`, and `prompt`, then executes returned actions.

## How to add a new custom robot

The cleanest way is to copy the `ump_suite_robot` pattern and replace only the robot-specific parts.

### Step 1: write a LeRobot conversion script

Create `examples/my_robot/convert_my_robot_data_to_lerobot.py`.

Use the ump suite converter as a template:

- choose a `REPO_NAME`
- define your dataset features
- map your raw logs into `image`, `state`, `actions`, and `task`
- call `ds.add_frame(...)` once per timestep
- call `ds.save_episode()` once per episode

Your demonstrations can come from CSV, HDF5, rosbags, RLDS, or anything else. The important part is the final LeRobot
dataset schema that `openpi` will train on.

### Step 2: write policy transforms

Create `src/openpi/policies/my_robot_policy.py`.

Use
[src/openpi/policies/ump_suite_robot_policy.py](../../src/openpi/policies/ump_suite_robot_policy.py)
as the template.

Your input transform should:

- read your observation keys
- convert images to `uint8` HWC if needed
- build the `image` dict expected by the model
- build the `image_mask` dict
- pass through `state`
- optionally pass through `actions` during training
- pass through `prompt`

Your output transform should:

- slice the model action output back to your robot's true action dimension

### Step 3: register a data config

In [src/openpi/training/config.py](../../src/openpi/training/config.py):

1. import your new policy module
2. add a `LeRobotMyRobotDataConfig`
3. map LeRobot dataset fields to inference-style keys with `RepackTransform`
4. attach your input/output policy transforms
5. add delta-action logic only if your raw actions are absolute and your model recipe expects deltas

The `LeRobotUmpSuiteRobotDataConfig` class is the reference example.

### Step 4: add train configs

Still in `config.py`, add one or more `TrainConfig` entries for your robot.

You typically need to set:

- `repo_id`
- model family (`pi0`, `pi0-FAST`, or `pi0.5`)
- `action_dim` for pi0-FAST
- `action_horizon`
- LoRA vs full finetune choice
- whether to reuse existing normalization stats or compute new ones

### Step 5: run the pipeline

```bash
uv run examples/my_robot/convert_my_robot_data_to_lerobot.py --data-root /path/to/raw
uv run scripts/compute_norm_stats.py --config-name pi0_my_robot
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_my_robot --exp-name=first_run --overwrite
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_my_robot \
    --policy.dir=checkpoints/pi0_my_robot/first_run/29999
```

### Step 6: connect your runtime

Your robot control loop should:

1. capture the current observation
2. format it exactly the way your policy input transform expects
3. send it to the websocket server
4. receive `actions`
5. execute those actions on the robot

If the training dataset keys and runtime inference keys stay semantically aligned, the rest of the openpi pipeline stays
fairly small and reusable.

## Practical checklist

- `REPO_NAME` in the converter matches `repo_id` in the training config.
- State dimension matches everywhere.
- Action dimension matches everywhere.
- The action ordering is identical in logs, conversion, transforms, and runtime.
- Episodes are at least as long as `action_horizon`.
- `scripts/compute_norm_stats.py` has been run for the config you plan to train.
- The robot runtime sends the same observation semantics that the training pipeline used.
