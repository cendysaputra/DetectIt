from pathlib import Path

import pytest
from PIL import Image

from detectit.data.dataset import SPLITS, check_records, load_manifest, prepare, stratify


def test_split_is_deterministic_disjoint_and_complete():
    records = [
        {"image": str(i), "classes": [i % 2] if i % 3 else [0, 1]}
        for i in range(30)
    ]
    ratios = [0.70, 0.15, 0.15]
    splits = stratify(records, ratios, seed=42)
    assert splits == stratify(records, ratios, seed=42)
    ids = [record["image"] for split in SPLITS for record in splits[split]]
    assert sorted(ids) == sorted({record["image"] for record in records})
    for split in SPLITS:
        assert {class_id for record in splits[split] for class_id in record["classes"]} == {0, 1}


def test_manifest_preserves_published_split_and_checks_hashes(tmp_path):
    for index, split in enumerate(("train", "valid", "test")):
        folder = tmp_path / split
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir()
        Image.new("RGB", (20, 20), (100 * index, 0, 0)).save(folder / "images" / "one.png")
        (folder / "labels" / "one.txt").write_text("0 .5 .5 .4 .4\n1 .2 .2 .2 .2\n")
    output = tmp_path / "splits.json"
    prepare(
        tmp_path, output, name="fixture", dataset_version="1", license_name="MIT",
        source="fixture", names=["helmet", "no helmet"], ratios=[0.70, 0.15, 0.15],
        seed=42, layout="published",
    )
    manifest = load_manifest(output)
    row = check_records(manifest, "val")[0]
    assert Path(row["image"]).parts == ("valid", "images", "one.png")
    (tmp_path / row["label"]).write_text("0 .5 .5 .4 .4\n")
    with pytest.raises(ValueError, match="Frozen data changed"):
        check_records(manifest, "val")


def test_ultralytics_can_modify_staging_without_changing_frozen_data(tmp_path):
    from detectit.train.finetune import training_data

    for index, split in enumerate(SPLITS):
        folder = tmp_path / "source" / split
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir()
        Image.new("RGB", (20, 20), (index * 90, 0, 0)).save(folder / "images" / "one.jpg")
        (folder / "labels" / "one.txt").write_text("0 .5 .5 .4 .4\n1 .2 .2 .2 .2\n")
    output = tmp_path / "splits.json"
    prepare(
        tmp_path / "source", output, name="fixture", dataset_version="1", license_name="MIT",
        source="fixture", names=["helmet", "no helmet"], ratios=[0.70, 0.15, 0.15],
        seed=42, layout="published",
    )
    manifest = load_manifest(output)
    run = tmp_path / "run"
    training_data(manifest, run)
    (run / "training-data" / "train" / "images" / "one.jpg").write_bytes(b"repaired")
    assert check_records(manifest, "train")
    assert not (run / "training-data" / "test").exists()
