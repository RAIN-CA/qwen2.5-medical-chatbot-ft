import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["train"])
    parser.add_argument("--model_family", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--run_type", type=str, required=True)
    args = parser.parse_args()

    if args.stage == "train":
        config_path = Path("configs/train") / args.model_family / args.model_size / f"{args.run_type}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        cmd = [
            sys.executable,
            "-m",
            "src.train.train_lora",
            "--config",
            str(config_path),
        ]
        run_command(cmd)


if __name__ == "__main__":
    main()
