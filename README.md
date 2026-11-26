# DetectIt

**See it, then box it.** Object detection with bounding boxes, class labels, and confidence.

## Overview

DetectIt is a learning and portfolio project for object detection, one step up from image classification. Classification names the thing in the picture. Detection also says where it is: a box, a class, and a confidence for each object.

The work starts from a pretrained Ultralytics YOLO model on the 80 COCO classes, draws those predictions, and then fine-tunes the same model on a small custom set. The custom set is an Indonesian street scene, motorbikes and helmets or a wider set of street vehicles. A Faster R-CNN baseline from torchvision is an optional comparison.

This repository currently holds the plan and the pipeline figure. The library, the training loop, the evaluation scripts, and the demo are still to be written. Any score that later appears in this file will come from a run recorded under `eval/`. The rules are in [PLAN.md](PLAN.md).

## How it works

![DetectIt pipeline](docs/pipeline.svg)

1. **Image/Video.** An uploaded image, a frame from a video file, or a webcam frame.
2. **Preprocess/Resize.** Decode the frame and resize it to the input size the checkpoint was trained with. The original size is kept, so the boxes can be drawn on the picture the person actually uploaded.
3. **Detector backbone + neck + head.** The backbone extracts features, the neck combines them across scales, and the head proposes class scores and boxes.
4. **Raw boxes.** The head's candidates, before filtering. Many of them overlap.
5. **Non-Maximum Suppression.** Remove candidates below the confidence threshold, then drop overlapping boxes of the same class whose overlap is above the IoU threshold.
6. **Boxes, labels, confidence.** The boxes the demo draws, and the boxes the per-class count table counts.

## Features

- Pretrained inference with an Ultralytics nano detector. YOLO11n is the default checkpoint. YOLOv8n is the same-size alternative, selected in config.
- Boxes drawn with the class name and the confidence.
- Confidence and IoU thresholds that the demo can move.
- A Streamlit app to upload an image or a video, inspect the boxes, and read a per-class count table. Webcam input is optional once upload works.
- Fine-tuning on a small custom dataset, starting from the pretrained weights.
- An optional Faster R-CNN comparison, using torchvision.
- An evaluation script that writes a log and a metrics file, so a number in this README can be traced to a command.

## Datasets

**COCO.** The pretrained checkpoints are Ultralytics models trained on [COCO](https://cocodataset.org). Detection uses the 80-class label set. The official splits stay as COCO published them. This repository does not redistribute COCO images.

**Custom set, Indonesian street context.** The fine-tune is meant to be small and specific. The first idea is motorbike helmets: a rider wearing a helmet, and a rider who is not. Motorbikes are ordinary in Indonesian traffic, and the two classes are easy to explain. A broader street-vehicle set (motorbike, car, bus, truck) is the fallback if helmet boxes are too small to label reliably.

Candidate public sets on Roboflow Universe, both using Indonesian class names for helmet use:

- [motocyclehelmet_detection](https://universe.roboflow.com/deteksihelm/motocyclehelmet_detection) (`pakai-helm`, `tidak-pakai-helm`)
- [Deteksi Helm Kendaraan Bermotor](https://universe.roboflow.com/abdul-jalil-azis/deteksi-helm-kendaraan-bermotor-pwech) (`pakai helm`, `tanpa helm`)

Images we label ourselves are the other source, if a public set's license or label quality does not fit. The chosen set, its version, and its license are written into the run config before training. Image files stay in `data/` and are not committed. The test split is fixed before training.

## Tech stack

- Python
- PyTorch
- Ultralytics
- OpenCV
- Streamlit
- torchvision, for the optional Faster R-CNN baseline

## Project structure

Planned. Today the tree is `README.md`, `PLAN.md`, and `docs/`.

```text
data/                 datasets and downloaded weights (gitignored)
notebooks/            box-format checks and sample drawings
src/detectit/
  data/               loading, splits, box-format conversion
  models/             YOLO wrapper and optional Faster R-CNN
  train/              fine-tuning loop
  eval/               metric helpers used by the scripts
  utils/              drawing, counts, seeding
eval/                 scripts that write logs and metrics JSON
app/                  Streamlit demo
configs/              one YAML per experiment
tests/
docs/                 pipeline.svg and pipeline.png
```

## Getting started

**Planned / work in progress.** Nothing here is runnable yet. The commands are the interface in [PLAN.md](PLAN.md). They will fail until the package, the configs, and the app exist.

### Install (planned)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[app,dev]"
```

### Inference (planned)

```bash
python -m detectit.predict --config configs/yolo11n-coco.yaml --source path/to/image.jpg
```

### Train (planned)

```bash
python eval/train.py --config configs/finetune-helmets.yaml
```

### Evaluate (planned)

```bash
python eval/evaluate.py --config configs/yolo11n-coco.yaml --split val
```

The script writes `eval/runs/<run-id>/metrics.json` together with the log, the resolved config, and the environment. A number enters this README only from that JSON.

### Run the app (planned)

```bash
streamlit run app/streamlit_app.py
```

## Results

No evaluation has been run. After a real run, this section will report mAP@0.5, mAP@0.5:0.95, precision, recall, per-class AP, and latency on CPU and, when a GPU was available, on GPU. Each figure will name the `eval/runs/` directory that produced it.

The comparison point for a pretrained checkpoint is the COCO mAP Ultralytics publishes for that checkpoint. The citation is in [PLAN.md](PLAN.md). Ultralytics published that figure. This repository has not measured it, so it is not listed here.

## Roadmap

1. Pretrained YOLO inference on images, with boxes, labels, and confidence.
2. An evaluation script, then a comparison with the published COCO mAP of the loaded checkpoint.
3. A small custom dataset, split into train, validation, and test before any training.
4. Fine-tuning, with every model choice made on validation.
5. The Streamlit demo: upload, thresholds, boxes, and a per-class count table.
6. Optional Faster R-CNN, and optional webcam input.

Tasks, metric definitions, and the evidence rule are in [PLAN.md](PLAN.md).

## Related projects

Sibling learning projects by [cendysaputra](https://github.com/cendysaputra):

- [VisionClassify](https://github.com/cendysaputra/VisionClassify) — CNN image classification on CIFAR-10, 95.71% test accuracy.
- [ExplainVision](https://github.com/cendysaputra/ExplainVision) — Grad-CAM explainability.
- [AudioScope](https://github.com/cendysaputra/AudioScope) — spectrogram sound classification.
- [VecScope](https://github.com/cendysaputra/VecScope) — embedding explorer.

## License

MIT license, placeholder. A `LICENSE` file will be added with the first code commit. Until then this repository is the plan and the pipeline figure.

The placeholder applies to original text and, later, original code in this repository. It does not cover COCO, a Roboflow dataset, or third-party weights. The Ultralytics package is GNU Affero General Public License v3. That has to be settled before the package is imported, and it will be written down in the commit that adds the dependency.
