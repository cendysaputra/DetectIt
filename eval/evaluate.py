import argparse
import sys
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path

from detectit.config import load_config
from detectit.eval.validation import evaluate
from detectit.utils.runs import Tee, create_run, git_state, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate a DetectIt checkpoint.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--device")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Score only a smoke subset.")
    parser.add_argument("--benchmark-images", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch, args.benchmark_images, args.limit if args.limit is not None else 1) < 1 or args.warmup < 0:
        parser.error("Batch, subset and benchmark sizes must be positive. Warmup must be nonnegative.")

    config = load_config(args.config)
    git_start = git_state()
    final_test = args.split == "test" and args.limit is None
    if args.split == "test" and (not final_test or not git_start[1]):
        parser.error("Test evaluation requires the full split and a clean, committed config.")
    lock = nullcontext()
    claim = None
    if final_test:
        revision_config = args.config.resolve()
        if not revision_config.is_relative_to(Path.cwd()) or not _tracked(revision_config):
            parser.error("The test config must be tracked in Git.")
        manifest_hash = sha256(config["dataset"]["manifest"])
        ledger = Path("data/test-evaluations")
        ledger.mkdir(parents=True, exist_ok=True)
        claim = ledger / f"{manifest_hash}.txt"
        try:
            lock = claim.open("x", encoding="utf-8")
        except FileExistsError:
            parser.error(f"This frozen test split already has a run recorded in {claim}.")

    with lock as record:
        try:
            run = create_run({
                **config, "config_source": str(args.config),
                "evaluation": {"split": args.split, "device": args.device, "batch": args.batch,
                               "limit": args.limit, "benchmark_images": args.benchmark_images, "warmup": args.warmup},
            })
            with (run / "eval.log").open("w", encoding="utf-8") as log, redirect_stdout(
                Tee(sys.stdout, log)
            ), redirect_stderr(Tee(sys.stderr, log)):
                print(f"Evaluating {args.config} into {run}", flush=True)
                metrics = evaluate(config, args, git_start)
                write_json(run / "metrics.json", metrics)
                print(f"Saved {run / 'metrics.json'} (reportable={metrics['reportable']})")
            if record:
                record.write(str(run) + "\n")
        except BaseException:
            if claim:
                claim.unlink(missing_ok=True)
            raise


def _tracked(path):
    import subprocess

    return subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", str(path)],
        capture_output=True, check=False,
    ).returncode == 0


if __name__ == "__main__":
    main()
