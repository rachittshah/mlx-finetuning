#!/usr/bin/env python3
"""
MLX Fine-tuning Script for GUI

This script performs LoRA fine-tuning using mlx-lm.
It's designed to be called by the GUI backend with a config file.
"""

import argparse
import yaml
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="MLX LoRA fine-tuning script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract parameters
    model_path = config["base_model_dir"]
    data_dir = config["prepared_data_dir"]
    adapter_path = Path(config["adapter_output_dir"]) / config["adapter_name"]

    # Prepare command-line arguments for mlx_lm.lora
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model_path,
        "--train",
        "--data", data_dir,
        "--adapter-path", str(adapter_path),
        "--iters", str(config["iters"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--steps-per-report", str(config["steps_per_report"]),
        "--steps-per-eval", str(config["steps_per_eval"]),
        "--save-every", str(config["save_every"]),
        "--max-seq-length", str(config.get("max_seq_length", 2048)),
    ]

    # Add optional parameters
    if config.get("val_batches", -1) > 0:
        cmd.extend(["--val-batches", str(config["val_batches"])])

    if config.get("resume_adapter_file"):
        cmd.extend(["--resume-adapter-file", config["resume_adapter_file"]])

    if config.get("grad_checkpoint", False):
        cmd.append("--grad-checkpoint")

    print(f"Starting MLX LoRA fine-tuning...")
    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Adapter output: {adapter_path}")
    print(f"Iterations: {config['iters']}")
    print(f"\nRunning command: {' '.join(cmd)}\n")

    try:
        # Run LoRA training via subprocess
        result = subprocess.run(cmd, check=True)
        print("\n* Training completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n* Training failed with exit code: {e.returncode}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"\n* Training failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
