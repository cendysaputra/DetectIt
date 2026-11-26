import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import yaml


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def git_state():
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None, False
    return revision, not changes


def is_reportable(*, full_split, clean_start, clean_end, same_revision, fixed_test=True):
    return all((full_split, clean_start, clean_end, same_revision, fixed_test))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def create_run(config):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = Path("eval/runs") / f"{timestamp}-{uuid4().hex[:8]}"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    packages = ("torch", "torchvision", "ultralytics", "opencv-python", "pycocotools", "numpy")
    environment = [f"Python {sys.version.split()[0]}"]
    environment.extend(f"{package}=={version(package)}" for package in packages)
    (run / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")
    return run


class Tee:
    def __init__(self, console, file):
        self.console = console
        self.file = file

    def write(self, message):
        self.console.write(message)
        return self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def isatty(self):
        return False
