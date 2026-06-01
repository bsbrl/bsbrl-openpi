"""Upload a local folder (a trained checkpoint or a dataset) to the Hugging Face Hub.

Examples:
    # Push a finetuned checkpoint (a model repo)
    uv run examples/sensapex/push_to_huggingface.py \
        --local-dir checkpoints/pi0_sensapex_low_mem_finetune/my_experiment/29999 \
        --repo-id RaianSilex/pi0_sensapex_low_mem_finetune \
        --repo-type model

    # Push a LeRobot dataset directory (a dataset repo)
    uv run examples/sensapex/push_to_huggingface.py \
        --local-dir ~/.cache/huggingface/lerobot/RaianSilex/ump_suite_robot_dataset \
        --repo-id RaianSilex/ump_suite_robot_dataset \
        --repo-type dataset

Note: the data converter (convert_data_to_lerobot.py) already pushes the dataset
when --push-to-hub is set, so this script is mainly useful for pushing trained
checkpoints after finetuning.
"""

import dataclasses

from huggingface_hub import HfApi
import tyro


@dataclasses.dataclass
class Args:
    # Local folder to upload (a checkpoint step dir, or a LeRobot dataset dir).
    local_dir: str = "checkpoints/pi0_sensapex_low_mem_finetune/my_experiment/29999"
    # Destination repo on the Hub.
    repo_id: str = "RaianSilex/pi0_sensapex_low_mem_finetune"
    # "model" for checkpoints, "dataset" for LeRobot datasets.
    repo_type: str = "model"
    # Whether the repo should be private.
    private: bool = False
    commit_message: str = "Upload from openpi"


def main(args: Args) -> None:
    api = HfApi()

    # 1. Create the repo if it does not exist.
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    # 2. Upload the folder contents to the repo.
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.local_dir} -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
