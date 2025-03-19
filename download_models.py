import argparse

from huggingface_hub import snapshot_download
import os


def download_models(repo_id, save_dir=None):
    # Download the entire repository
    save_dir = os.path.abspath(save_dir)
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=save_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--save_dir', type=str, default="source/models")
    args = args.parse_args()

    repo_id = "chanyikchong/Early-Exit-Neural-Network"
    download_models(repo_id, args.save_dir)
