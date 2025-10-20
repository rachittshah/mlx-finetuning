#!/usr/bin/env python3
"""
Download MLX-compatible models from Hugging Face Hub

Usage:
    python download_model.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    python download_model.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
    python download_model.py --list-popular
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import sys

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODELS_DIR = PROJECT_ROOT / "artifacts" / "base_model"

# Popular MLX models (pre-quantized for Apple Silicon)
POPULAR_MODELS = {
    "qwen2.5-7b-4bit": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5-3b-4bit": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mistral-7b-4bit": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "llama3.2-3b-4bit": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "phi-3-4bit": "mlx-community/Phi-3-mini-4k-instruct-4bit",
    "gemma-2-2b-4bit": "mlx-community/gemma-2-2b-it-4bit",
}


def list_popular_models():
    """Display list of popular pre-configured models"""
    print("\n* Popular MLX Models (4-bit quantized for efficiency):\n")
    for name, repo in POPULAR_MODELS.items():
        print(f"  {name:20s} → {repo}")
    print("\nUsage:")
    print("  python download_model.py --model qwen2.5-7b-4bit")
    print("  python download_model.py --model mlx-community/Qwen2.5-7B-Instruct-4bit")
    print("\nBrowse all MLX models: https://huggingface.co/models?library=mlx")


def download_model(model_id: str, output_name: str = None):
    """
    Download a model from Hugging Face Hub

    Args:
        model_id: Either a shorthand name from POPULAR_MODELS or full repo ID
        output_name: Optional custom name for the downloaded model directory
    """
    # Check if it's a shorthand name
    if model_id in POPULAR_MODELS:
        print(f"📦 Using popular model: {model_id}")
        repo_id = POPULAR_MODELS[model_id]
        if output_name is None:
            output_name = model_id
    else:
        repo_id = model_id
        if output_name is None:
            # Extract model name from repo_id
            output_name = repo_id.split("/")[-1]

    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Local path for the model
    local_dir = MODELS_DIR / output_name

    # Check if already downloaded
    if local_dir.exists() and (local_dir / "config.json").exists():
        print(f"* Model already exists at: {local_dir}")
        response = input("Download again? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return

    print(f"\n* Downloading model: {repo_id}")
    print(f"📁 Saving to: {local_dir}\n")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        print(f"\n* Successfully downloaded model to: {local_dir}")
        print(f"\n* Model is now available in the GUI!")

        # Verify essential files
        config_file = local_dir / "config.json"
        if config_file.exists():
            print(f"✓ config.json found")

        weights_files = list(local_dir.glob("*.safetensors"))
        if weights_files:
            print(f"✓ {len(weights_files)} weight file(s) found")

    except Exception as e:
        print(f"\n* Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download MLX-compatible models from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using shorthand name
  python download_model.py --model qwen2.5-7b-4bit

  # Download using full Hugging Face repo ID
  python download_model.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

  # List popular models
  python download_model.py --list-popular

  # Download with custom name
  python download_model.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --name my-mistral
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model ID (shorthand or full repo ID like 'mlx-community/Qwen2.5-7B-Instruct-4bit')"
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Custom name for the model directory (optional)"
    )

    parser.add_argument(
        "--list-popular",
        action="store_true",
        help="List popular pre-configured models"
    )

    args = parser.parse_args()

    if args.list_popular:
        list_popular_models()
    elif args.model:
        download_model(args.model, args.name)
    else:
        parser.print_help()
        print("\n")
        list_popular_models()


if __name__ == "__main__":
    main()
