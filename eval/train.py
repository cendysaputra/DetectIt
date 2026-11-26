import argparse
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from detectit.config import load_config
from detectit.train.finetune import finetune
from detectit.utils.runs import Tee, create_run, git_state


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on a frozen local dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = Path(config["dataset"]["manifest"])
    if not manifest.is_file():
        parser.error(f"Dataset manifest {manifest} is missing. Run python -m detectit.data.prepare first.")
    git_start = git_state()
    run = create_run({
        **config, "config_source": str(args.config),
        "runtime": {"device": args.device, "workers": args.workers},
    })
    with (run / "train.log").open("w", encoding="utf-8") as log, redirect_stdout(
        Tee(sys.stdout, log)
    ), redirect_stderr(Tee(sys.stderr, log)):
        print(f"Fine-tuning {args.config} into {run}", flush=True)
        finetune(config, run, args.device, args.workers, git_start)


if __name__ == "__main__":
    main()
