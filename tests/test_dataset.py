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


def test_followup_retains_prior_validation_and_uses_unseen_test_images(tmp_path):
    for split_index, split in enumerate(SPLITS):
        folder = tmp_path / split
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir()
        for index in range(6):
            Image.new("RGB", (20, 20), (split_index * 90, 10 * index, 0)).save(folder / "images" / f"{index}.png")
            (folder / "labels" / f"{index}.txt").write_text("0 .5 .5 .4 .4\n1 .2 .2 .2 .2\n")

    settings = dict(name="fixture", dataset_version="1", license_name="MIT", source="fixture",
                    names=["a", "b"], ratios=[0.70, 0.15, 0.15], seed=42, layout="published")
    original = tmp_path / "original.json"
    updated = tmp_path / "updated.json"
    prepare(tmp_path, original, sample_sizes={"train": 2, "val": 2, "test": 2}, **settings)
    prepare(tmp_path, updated, sample_sizes={"train": 4, "val": 2, "test": 4}, previous_manifest=original, **settings)
    first = load_manifest(original)["splits"]
    second = load_manifest(updated)["splits"]

    def hashes(records):
        return {record["image_sha256"] for record in records}

    assert first["val"] == second["val"]
    assert hashes(first["train"]) < hashes(second["train"])
    assert hashes(first["test"]).isdisjoint(hashes(second["test"]))


def test_targeted_sampling_keeps_all_prior_test_images_out(tmp_path):
    for split_index, split in enumerate(SPLITS):
        folder = tmp_path / split
        (folder / "images").mkdir(parents=True)
        (folder / "labels").mkdir()
        for index in range(15):
            stem = f"{index:02}"
            Image.new("RGB", (20, 20), (split_index * 90, 10 * index, 0)).save(folder / "images" / f"{stem}.png")
            label = "0 .5 .5 .4 .4\n"
            if index % 3 == 0:
                label += "1 .2 .2 .2 .2\n"
            (folder / "labels" / f"{stem}.txt").write_text(label)

    settings = dict(name="fixture", dataset_version="1", license_name="MIT", source="fixture",
                    names=["a", "b"], ratios=[0.70, 0.15, 0.15], seed=42, layout="published")
    first_path, second_path, third_path = [tmp_path / f"v{index}.json" for index in (1, 2, 3)]
    prepare(tmp_path, first_path, sample_sizes={"train": 2, "val": 2, "test": 2}, **settings)
    prepare(tmp_path, second_path, sample_sizes={"train": 3, "val": 2, "test": 2},
            previous_manifest=first_path, **settings)
    prepare(tmp_path, third_path, sample_sizes={"train": 6, "val": 2, "test": 3},
            previous_manifest=second_path, additional_test_exclusions=[first_path],
            extra_train_minimums={1: 2}, **settings)
    first, second, third = [load_manifest(path)["splits"] for path in (first_path, second_path, third_path)]
    earlier_test = {row["image_sha256"] for split in (first, second) for row in split["test"]}
    assert earlier_test.isdisjoint(row["image_sha256"] for row in third["test"])
    earlier_train = {row["image_sha256"] for row in second["train"]}
    added = [row for row in third["train"] if row["image_sha256"] not in earlier_train]
    assert len(added) == 3
    assert sum(1 in row["classes"] for row in added) >= 2
    assert first["val"] == third["val"]
