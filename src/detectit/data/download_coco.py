import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

BASE_URL = "https://images.cocodataset.org"


def download(root):
    images = root / "images" / "val2017"
    annotations = root / "annotations" / "instances_val2017.json"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / "download.zip"
        if not images.exists():
            print("Downloading COCO val2017 images", flush=True)
            urlretrieve(f"{BASE_URL}/zips/val2017.zip", archive_path)
            with ZipFile(archive_path) as archive:
                archive.extractall(temporary)
            images.parent.mkdir(parents=True, exist_ok=True)
            (temporary / "val2017").rename(images)
        if not annotations.exists():
            print("Downloading COCO 2017 annotations", flush=True)
            urlretrieve(f"{BASE_URL}/annotations/annotations_trainval2017.zip", archive_path)
            extracted = temporary / annotations.name
            with ZipFile(archive_path) as archive, archive.open("annotations/instances_val2017.json") as source:
                with extracted.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            annotations.parent.mkdir(parents=True, exist_ok=True)
            extracted.rename(annotations)
    print(f"COCO val2017 ready under {root}")


def main():
    parser = argparse.ArgumentParser(description="Download official COCO val2017 images and annotations.")
    parser.add_argument("--root", type=Path, default=Path("data/coco"))
    download(parser.parse_args().root)


if __name__ == "__main__":
    main()
