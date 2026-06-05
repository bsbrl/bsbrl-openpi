# Fine-tuning π₀ (LoRA, low-mem) on MSI — HuggingFace dataset → checkpoint

Minimal, copy-paste recipe. Trains the **`pi0_sensapex_low_mem_finetune`** config (pi0 + LoRA,
EMA off, action_horizon=10, action_dim=8→32) from a **LeRobot v2.1** dataset on HuggingFace.
To upload a v2.1 dataset first, see [MicroVLA/HUGGINGFACE_LEROBOT_UPLOAD.md](../../MicroVLA/HUGGINGFACE_LEROBOT_UPLOAD.md).

> **Placeholders to change before pasting:**
> - `chowd207@login.msi.umn.edu` → your MSI login
> - `/projects/standard/suhasabk/shared` → your MSI project dir
> - `RaianSilex/multibead_165episodes_lerobotv21` → your **v2.1** HF dataset repo id

---

## 0. The rules that bit us

| Rule | Why |
|---|---|
| **Never commit an HF token** (no `export HF_TOKEN=hf_…` in any file) | Push protection blocks it; it leaks. Use `huggingface-cli login` (token cached in `HF_HOME`). |
| **Never rsync `.venv/` or `.cache/`** | `.venv` is ~8–12 GB; `--delete` wipes caches. |
| **Pre-download pi0_base + dataset on the LOGIN node** | Compute nodes have **no internet**; runtime downloads hang there. |
| **`uv sync` with `--no-install-package rerun-sdk`** | `rerun-sdk` has no manylinux_2_28 wheel (MSI is glibc 2.28). |
| **Set `UV_NO_SYNC=1` in the job** | Stops `uv run` from touching the internet on the compute node. |
| **Resume with `--resume`, never `--overwrite`** | `--overwrite` wipes the run to step 0. |

---

## 1. Sync code to MSI (from your laptop)

```bash
cd ~/bsbrl-openpi
rsync -avhP --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '.cache/' \
  --exclude '__pycache__/' --exclude '*/__pycache__/' \
  --exclude 'logs/' --exclude 'checkpoints/' --exclude 'wandb/' --exclude 'assets/' \
  ~/bsbrl-openpi/ chowd207@login.msi.umn.edu:/projects/standard/suhasabk/shared/bsbrl-openpi/
```

---

## 2. Point the config at YOUR dataset

`src/openpi/training/config.py` ships pointing at the old 137-episode set. Repoint **all**
`LeRobotSensapexDataConfig` repo_ids to your v2.1 dataset (run on MSI, in the repo):

```bash
cd /projects/standard/suhasabk/shared/bsbrl-openpi
sed -i 's|RaianSilex/ump_robot_dataset_137_episodes_lerobotv21|RaianSilex/multibead_165episodes_lerobotv21|g' \
  src/openpi/training/config.py     # CHANGE the second repo id to yours
grep -n "repo_id=\"RaianSilex" src/openpi/training/config.py   # verify all lines now show your dataset
```
> `compute_norm_stats` and `train.py` both read the dataset **from `config.py`**, not from any
> manual download. If this is wrong, you compute/train on the wrong data.

---

## 3. One-time on MSI (LOGIN node): env + auth + downloads

```bash
ssh chowd207@login.msi.umn.edu
cd /projects/standard/suhasabk/shared/bsbrl-openpi
export HF_HOME=$PWD/.cache/huggingface       # keep this same value everywhere (login + job)

# 3a. Build the env from the lockfile, minus rerun (no glibc-2.28 wheel). ~8–12 GB; do NOT `uv pip install -e .`
uv sync --frozen --no-install-package rerun-sdk

# 3b. HF auth — paste a READ token (hidden). Token is cached under $HF_HOME; no token in any file.
uv run huggingface-cli login

# 3c. Pre-download pi0_base (~10+ GB) via gsutil (login node has internet)
which gsutil || uv pip install gsutil
uv run python -c "import openpi.shared.download as d; print(d.maybe_download('gs://openpi-assets/checkpoints/pi0_base/params'))"

# 3d. Pre-download the dataset into HF_HOME so the compute node doesn't need internet
uv run python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('RaianSilex/multibead_165episodes_lerobotv21')"   # CHANGE to your repo id
```

---

## 4. Compute norm stats (LOGIN node, after §2)

Must use the **exact training config name** — it writes to `assets/<config>/<repo_id>/`:
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_sensapex_low_mem_finetune
```
Stats are written only at the very end, so an interrupted run leaves nothing; just re-run. It's
idempotent (overwrites). Changing the dataset (§2) means re-running this.

---

## 5. Smoke test (LOGIN node or a short interactive GPU)

```bash
export HF_HOME=$PWD/.cache/huggingface XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 UV_NO_SYNC=1
uv run scripts/train.py pi0_sensapex_low_mem_finetune \
  --exp-name=smoke --overwrite --no-wandb-enabled \
  --num-train-steps=10 --batch-size=16 --num-workers=8
```
Loss dropping over 10 steps = good. (First step is slow: XLA compile.)

---

## 6. The sbatch (full 30k run)

Save as `train_pi0_lora.sbatch`. **No token anywhere** — it relies on the cached login from §3b.

```bash
#!/bin/bash -l
#SBATCH --job-name=pi0-sensapex-lowmem
#SBATCH -p msigpu
#SBATCH --gres=gpu:h100:1            # CHANGE: a100:1 also works (40/80 GB)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH --tmp=200g
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chowd207@umn.edu   # CHANGE

set -euo pipefail
cd /projects/standard/suhasabk/shared/bsbrl-openpi    # CHANGE
mkdir -p logs

export HF_HOME=$PWD/.cache/huggingface     # same value used at `huggingface-cli login` (§3b)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export UV_NO_SYNC=1                          # no internet calls on the compute node

echo "Node: $(hostname)  Start: $(date)"; nvidia-smi

uv run scripts/train.py pi0_sensapex_low_mem_finetune \
  --exp-name=run1 \
  --overwrite \                              # CHANGE to --resume to continue a killed run (never both)
  --num-train-steps=30000 \
  --batch-size=32 --num-workers=8 \
  --save-interval=2000 --keep-period=10000
echo "End: $(date)"
```

Submit + watch (**progress is in `.err`** — tqdm `Progress on: X/30.0kit`):
```bash
cd /projects/standard/suhasabk/shared/bsbrl-openpi
sbatch train_pi0_lora.sbatch
squeue --me
tail -f logs/pi0-sensapex-lowmem-*.err
```

> **Timing:** ~2.6 s/it on an A100 → ~21–22 h for 30k steps; fits a 24 h wall but it's tight.
> Checkpoints save every `--save-interval` under `checkpoints/<config>/run1/`. If the wall hits first,
> resubmit with `--overwrite`→`--resume` to continue from the last checkpoint.

---

## 7. Change the dataset / config / steps

- **Dataset:** redo §2 (sed `config.py`) + §4 (recompute norm stats). Must be **LeRobot v2.1**.
- **Config:** swap `pi0_sensapex_low_mem_finetune` everywhere (norm stats, smoke, sbatch) for another
  config name in `config.py` (e.g. full `pi0_sensapex`). Norm stats are keyed to the config.
- **Steps / batch:** `--num-train-steps`, `--batch-size` on the train command.

---

## 8. Gotchas

| Symptom | Fix |
|---|---|
| GitHub push blocked (secret) | a token is in history; never write `export HF_TOKEN=hf_…` — use `huggingface-cli login` |
| `uv pip install -e .` fails on `rerun-sdk` | use `uv sync --frozen --no-install-package rerun-sdk`; don't `pip install -e .` |
| job hangs, GPU idle, no progress | runtime download on a no-internet node → pre-download pi0_base + dataset (§3c/§3d) |
| "Missing norm stats" at start | skipped §4, or config name mismatch → recompute with the **exact** training config |
| computed stats on the wrong dataset | `config.py` repo_id not repointed (§2) — sed + verify with grep, then recompute |
| OOM on 40 GB A100 | keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`; drop `--batch-size` to 16 or 8 |
| run restarts from step 0 on resubmit | you used `--overwrite`; switch to `--resume` |