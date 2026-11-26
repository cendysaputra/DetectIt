import argparse
from pathlib import Path

from detectit.config import load_config
from detectit.data.dataset import prepare


def main():
    parser = argparse.ArgumentParser(description="Freeze a local YOLO detection dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/finetune-helmets.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--names", nargs="+", required=True, help="Class names in label ID order.")
    parser.add_argument("--layout", choices=("published", "standard", "unsplit"), default="published")
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["dataset"]["manifest"])
    prepare(
        args.root, output, name=args.name, dataset_version=args.version,
        license_name=args.license, source=args.source, names=args.names, layout=args.layout,
        ratios=config["dataset"]["ratios"], seed=config["seed"],
        sample_sizes=config["dataset"].get("sample_sizes"),
    )
    print(f"Frozen split manifest: {output}")


if __name__ == "__main__":
    main()
