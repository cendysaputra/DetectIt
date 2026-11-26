import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from pycocotools.coco import COCO

from detectit.data.boxes import clip_xyxy, normalized_cxcywh_to_xyxy, xyxy_to_xywh
from detectit.utils.runs import sha256, write_json

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def read_labels(path, class_count):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = line.split()
        class_id = int(class_id)
        box = tuple(map(float, (x, y, width, height)))
        if not 0 <= class_id < class_count or not all(0 <= value <= 1 for value in box) or min(box[2:]) <= 0:
            raise ValueError(f"Invalid YOLO detection annotation in {path}: {line}")
        labels.append((class_id, box))
    return labels


def stratify(records, ratios, seed):
    if len(ratios) != len(SPLITS) or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Provide three positive split ratios that sum to 1.")
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record["classes"])].append(record)
    rng = random.Random(seed)
    assignments = {key: [] for key in SPLITS}
    global_targets = [len(records) * ratio for ratio in ratios]
    for classes in sorted(groups, key=lambda key: (len(groups[key]), key)):
        group = groups[classes]
        rng.shuffle(group)
        group_targets = [len(group) * ratio for ratio in ratios]
        if len(group) >= len(SPLITS) and classes:
            for index, split in enumerate(SPLITS):
                assignments[split].append(group[index])
                group_targets[index] -= 1
                global_targets[index] -= 1
            group = group[len(SPLITS):]
        for record in group:
            index = max(range(len(SPLITS)), key=lambda i: (group_targets[i], global_targets[i]))
            assignments[SPLITS[index]].append(record)
            group_targets[index] -= 1
            global_targets[index] -= 1
    return assignments


def paired_images(root, folder):
    images = folder / "images"
    for path in sorted(images.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            label = folder / "labels" / path.relative_to(images).with_suffix(".txt")
            yield path.relative_to(root), label.relative_to(root)


def sample_split(records, size, seed):
    if not 0 < size <= len(records):
        raise ValueError(f"Sample size {size} exceeds the available {len(records)} images or is not positive.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    uncovered = {class_id for record in records for class_id in record["classes"]}
    sample = []
    while uncovered:
        best = max(shuffled, key=lambda record: len(uncovered.intersection(record["classes"])))
        shuffled.remove(best)
        sample.append(best)
        uncovered.difference_update(best["classes"])
    if len(sample) > size:
        raise ValueError("The requested sample is too small to preserve all classes.")
    return sample + shuffled[:size - len(sample)]


def extend_training(records, size, seed, minimums):
    if not minimums:
        return sample_split(records, size, seed)
    rng = random.Random(seed)
    available = list(records)
    rng.shuffle(available)
    selected = []
    for class_id, minimum in minimums.items():
        present = sum(class_id in record["classes"] for record in selected)
        matches = [record for record in available if class_id in record["classes"]]
        needed = max(0, minimum - present)
        if needed > len(matches):
            raise ValueError(f"Not enough additional training images for class {class_id}.")
        chosen = matches[:needed]
        selected.extend(chosen)
        chosen_hashes = {record["image_sha256"] for record in chosen}
        available = [record for record in available if record["image_sha256"] not in chosen_hashes]
    remaining = size - len(selected)
    if not 0 <= remaining <= len(available):
        raise ValueError("Training sample size cannot accommodate the configured class minimums.")
    return selected + available[:remaining]


def prepare(root, output, *, name, dataset_version, license_name, source, names, ratios, seed, layout,
            sample_sizes=None, previous_manifest=None, additional_test_exclusions=(), extra_train_minimums=None):
    root = root.resolve()
    if not names or len(set(names)) != len(names) or len(names) < 2:
        raise ValueError("Provide at least two distinct class names in label ID order.")
    if output.exists():
        raise FileExistsError(f"{output} is already frozen. Use a new dataset version and output path.")
    candidates = {}
    if layout == "published":
        for split in SPLITS:
            folder = root / split
            if split == "val" and not folder.exists():
                folder = root / "valid"
            candidates[split] = list(paired_images(root, folder))
    elif layout == "standard":
        for split in SPLITS:
            images = root / "images" / split
            candidates[split] = [
                (path.relative_to(root), (root / "labels" / split / path.relative_to(images)).relative_to(root).with_suffix(".txt"))
                for path in sorted(images.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
    else:
        candidates["all"] = list(paired_images(root, root))

    splits = {}
    image_hashes = Counter()
    seen_labels = set()
    for split, pairs in candidates.items():
        records = []
        for image, label in pairs:
            labels = read_labels(root / label, len(names))
            image_hash = sha256(root / image)
            if label in seen_labels:
                raise ValueError(f"Multiple images would share one label file: {image}")
            image_hashes[image_hash] += 1
            seen_labels.add(label)
            records.append({
                "image": image.as_posix(), "label": label.as_posix(),
                "image_sha256": image_hash, "label_sha256": sha256(root / label),
                "classes": sorted({class_id for class_id, _ in labels}),
            })
        splits[split] = records

    if layout == "unsplit":
        splits = stratify(splits["all"], ratios, seed)
    source_sizes = {split: len(records) for split, records in splits.items()}
    duplicate_images = sum(count for count in image_hashes.values() if count > 1)
    if duplicate_images and sample_sizes is None:
        raise ValueError("The source contains byte-identical images. Resolve duplicates before freezing the splits.")
    if previous_manifest is not None and (sample_sizes is None or layout == "unsplit"):
        raise ValueError("A previous manifest requires sampling from published splits.")
    previous = load_manifest(previous_manifest) if previous_manifest is not None else None
    if (additional_test_exclusions or extra_train_minimums) and previous is None:
        raise ValueError("Additional test exclusions and training minimums require a previous manifest.")
    excluded_test_hashes = {
        record["image_sha256"]
        for path in additional_test_exclusions
        for record in load_manifest(path)["splits"]["test"]
    }
    if previous is not None and previous["names"] != names:
        raise ValueError("Class names must match the previous manifest.")
    if sample_sizes is not None:
        for split in SPLITS:
            unique = [record for record in splits[split] if image_hashes[record["image_sha256"]] == 1]
            if previous is None:
                splits[split] = sample_split(unique, sample_sizes[split], seed)
                continue
            earlier = previous["splits"][split]
            earlier_hashes = {record["image_sha256"] for record in earlier}
            extra = [record for record in unique if record["image_sha256"] not in earlier_hashes]
            if split == "test":
                unseen = [record for record in extra if record["image_sha256"] not in excluded_test_hashes]
                splits[split] = sample_split(unseen, sample_sizes[split], seed)
                continue
            retained = [record for record in unique if record["image_sha256"] in earlier_hashes]
            if retained != earlier:
                raise ValueError(f"The previous {split} records no longer match the dataset.")
            increase = sample_sizes[split] - len(retained)
            if increase < 0 or (split == "val" and increase):
                raise ValueError("Keep the previous validation split and retain all previous training images.")
            minimums = extra_train_minimums if split == "train" else None
            if increase or minimums:
                retained += extend_training(extra, increase, seed, minimums)
            splits[split] = retained
        print(f"Excluded {duplicate_images} source images with byte-identical copies before sampling.")
    for split, records in splits.items():
        counts = Counter(
            names[class_id] for record in records
            for class_id, _ in read_labels(root / record["label"], len(names))
        )
        class_counts = {name: counts[name] for name in names}
        print(f"{split}: {len(records)} images, {class_counts}")
        if not all(class_counts.values()):
            raise ValueError(f"{split} is missing at least one class.")
        records.sort(key=lambda record: record["image"])

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root": os.path.relpath(root, output.parent.resolve()),
        "name": name, "version": dataset_version, "license": license_name, "source": source,
        "names": names, "layout": layout, "seed": seed,
        "ratios": ratios if layout == "unsplit" else None, "source_sizes": source_sizes,
        "excluded_duplicate_images": duplicate_images, "splits": splits,
    }
    if previous_manifest is not None:
        manifest["previous_manifest_sha256"] = sha256(previous_manifest)
    if additional_test_exclusions:
        manifest["additional_test_exclusion_sha256"] = [sha256(path) for path in additional_test_exclusions]
    if extra_train_minimums:
        manifest["extra_train_minimums"] = extra_train_minimums
    write_json(output, manifest)


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["root"] = (path.resolve().parent / manifest["root"]).resolve()
    image_ids = set()
    for split in SPLITS:
        records = manifest["splits"][split]
        classes = {class_id for record in records for class_id in record["classes"]}
        if classes != set(range(len(manifest["names"]))):
            raise ValueError(f"{split} is missing at least one class.")
        for record in records:
            image_hash = record["image_sha256"]
            if image_hash in image_ids:
                raise ValueError(f"Image is repeated in the manifest: {record['image']}")
            image_ids.add(image_hash)
    return manifest


def check_records(manifest, split):
    root = manifest["root"]
    records = manifest["splits"][split]
    for record in records:
        for key in ("image", "label"):
            path = root / record[key]
            if sha256(path) != record[f"{key}_sha256"]:
                raise ValueError(f"Frozen data changed: {path}")
    return records


def custom_ground_truth(manifest, split):
    root = manifest["root"]
    names = manifest["names"]
    images, annotations, paths = [], [], {}
    for image_id, record in enumerate(check_records(manifest, split), start=1):
        path = root / record["image"]
        paths[image_id] = path
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        images.append({"id": image_id, "file_name": record["image"], "width": width, "height": height})
        for class_id, box in read_labels(root / record["label"], len(names)):
            xyxy = clip_xyxy(normalized_cxcywh_to_xyxy(box, width, height), width, height)
            bbox = xyxy_to_xywh(xyxy)
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": class_id + 1,
                "bbox": list(bbox), "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {}, "images": images, "annotations": annotations,
        "categories": [{"id": index, "name": name} for index, name in enumerate(names, start=1)],
    }
    ground_truth.createIndex()
    return ground_truth, paths
