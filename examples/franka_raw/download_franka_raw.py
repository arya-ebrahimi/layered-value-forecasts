"""
Download raw Franka hdf5 episode files (one `episode_*.hdf5` per demo) from the Hugging Face Hub.

Reads the HF token from the `HF_TOKEN` env var (huggingface_hub picks it up automatically).

Example usage:
  uv run examples/franka_raw/download_franka_raw.py --repo-id <hf-user>/<dataset> --local-dir $SCRATCH/datasets/franka_raw_hdf5
"""

from pathlib import Path

from huggingface_hub import snapshot_download
import tyro


def main(local_dir: Path, repo_id: str):
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=["*.hdf5"],
    )
    n_files = len(list(Path(path).glob("episode_*.hdf5")))
    print(f"Downloaded {n_files} episode files to {path}")


if __name__ == "__main__":
    tyro.cli(main)
