import argparse
import os
from huggingface_hub import snapshot_download

def download_repo(repo_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    
    # Enable hf_transfer for faster downloads
    # If you already set HF_HUB_ENABLE_HF_TRANSFER in your environment, you can omit this line.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    print(f"Downloading repository '{repo_id}' to '{local_dir}' using hf_transfer...")
    local_path = snapshot_download(repo_id=repo_id, cache_dir=local_dir)
    print(f"Download complete. Files saved to: {local_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face repository to a specified local directory."
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="togethercomputer/phoenix-1layer-baseline",
        help="Hugging Face model repository ID to download (default: togethercomputer/phoenix-1layer-baseline)."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="/data/jamesliu/models",
        help="Local directory where the repository will be downloaded (default: /data/jamesliu/models)."
    )
    args = parser.parse_args()
    download_repo(args.repo, args.local_dir)

if __name__ == "__main__":
    main()