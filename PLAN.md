# DetectIt project plan

This document is the working plan for DetectIt. [README.md](README.md) is the public summary. It receives a measured result only after the evaluation rules below are met. The pipeline figure in the README is `docs/pipeline.svg` (with `docs/pipeline.png` beside it).

Checked tasks in [Milestones](#milestones) are done in the repository. Everything else is still ahead.

## Goals

- Run a pretrained Ultralytics YOLO detector on an image and draw each box with its class label and confidence.
- Fine-tune that detector on a small custom dataset with an Indonesian street context: motorbike helmets, or street vehicles if helmet labels do not hold up.
- Evaluate with the standard detection metrics, and compare the pretrained checkpoint with the COCO mAP Ultralytics publishes for it.
- Ship a Streamlit demo where a person uploads an image or a video, moves the confidence and IoU thresholds, and sees the boxes plus a per-class count table.
- Keep every DetectIt number in the README reproducible from a script in `eval/`, with the log, the metrics JSON, and the model, config, and dataset versions saved beside it.
- Leave the code readable enough to sit next to VisionClassify, ExplainVision, AudioScope, and VecScope.

## Non-goals

- A production surveillance system, a traffic-enforcement product, or a hosted deployment in the first completed version.
- Instance segmentation, pose, or oriented boxes. The task is axis-aligned detection.
- Multi-object tracking. A video is a sequence of frames. The count table describes the frame on screen. Summing those counts across frames counts the same rider again on each frame, and the demo will say so if it shows a clip total.
- Training a detector from random initialization on COCO. The project starts from published pretrained weights.
- Beating a public leaderboard. Published numbers are comparison points, cited from the page that printed them.
- Redistributing COCO, a Roboflow export, or pretrained weights. Those files stay local and keep their own licenses.
- Choosing a model, a threshold, or an augmentation setting on the test split.

## Background

These are the definitions the code will follow. Training, evaluation, and the demo use the same ones, so a box that is correct in a notebook is the box the app draws.

### Bounding box formats

A box is four numbers, and the four numbers are meaningless until the format is named. DetectIt will meet three formats:

| Format | Fields | Where it shows up |
| --- | --- | --- |
| `xyxy` | `x_min`, `y_min`, `x_max`, `y_max` | torchvision detection, including Faster R-CNN. Absolute pixels. |
| `xywh` (top-left) | `x_min`, `y_min`, `width`, `height` | COCO JSON annotations. Absolute pixels. |
| `cxcywh` (normalized) | `x_center`, `y_center`, `width`, `height` | Ultralytics YOLO label files. Each value is divided by the image width or height, so the numbers lie in `[0, 1]`. |

Conversion is a function with the source format and the destination format in its name. A round trip on a fixed box is a unit test. Mixing `xywh` with `xyxy`, or forgetting to multiply a normalized YOLO box by the image size, is a silent bug: the box is still four positive numbers, and it is in the wrong place.

The drawing code converts to absolute `xyxy` in the original image, then clips to the image bounds. The metric code uses the same absolute `xyxy` boxes, so a box that looks right is the box that is scored.

### Intersection over Union (IoU)

IoU is the area shared by two boxes divided by the area of their union. It is 0 when the boxes do not touch and 1 when they are the same box. Two uses are easy to confuse, and the code will keep them apart:

- **Matching IoU.** During evaluation, a prediction is a candidate true positive when its IoU with a ground-truth box of the same class is at least the matching threshold (0.50, or a higher COCO threshold). Each ground-truth box is matched to at most one prediction.
- **NMS IoU.** During inference, two predictions of the same class that overlap more than the NMS threshold are treated as the same object, and the lower-confidence one is removed.

The matching threshold is a property of the metric. The NMS threshold is a property of the detector's post-processing. They can both be 0.50 and still mean different things. The metrics JSON records both.

### Anchors and anchor-free heads

An anchor-based detector lays down a set of prior boxes at each location on the feature map and predicts a correction to each prior. Faster R-CNN is the anchor-based model in this plan: a region-proposal network scores anchors, and a second head classifies the proposals.

An anchor-free head does not keep that bank of prior boxes. It scores locations on the feature map and predicts a box from those locations directly. Ultralytics documents the YOLOv8 detect head as anchor-free ([YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/)). YOLO11 is the later Ultralytics detector this project loads first. Its model page describes a newer backbone and neck and does not repeat the anchor-free sentence. The checkpoint config saved with each run is the record of the head that was actually loaded. The Faster R-CNN run is what makes the anchor-based comparison concrete.

### Non-maximum suppression (NMS)

The head emits many overlapping boxes for one object. NMS sorts them by confidence and walks the list. It keeps a box, then removes later boxes of the same class whose IoU with the kept box is above the NMS threshold. Boxes of different classes are left alone, so a person and a motorbike can occupy the same region.

The demo's IoU slider is this NMS threshold. It is recorded in the config of any run whose precision and recall depend on it.

### Confidence threshold

Each box carries a confidence, a score the model assigns to that class at that box. The confidence threshold drops boxes below a chosen score. It is an operating point: a higher threshold usually raises precision and lowers recall.

It is the wrong knob for headline mAP. Average precision is computed from the precision-recall curve across scores. Ultralytics recommends leaving the validation confidence at the library's low default (`0.001`) so that curve is complete, and reporting precision and recall separately at the operating point used in the demo ([YOLO11 docs, evaluation note](https://docs.ultralytics.com/models/yolo11/)). DetectIt will follow that split:

- mAP@0.5 and mAP@0.5:0.95 come from the standard validation settings.
- Precision and recall are reported at the confidence the demo is using, and the JSON names that confidence.

### Precision and recall

At a chosen operating point, after NMS:

- **Precision** is the fraction of predicted boxes that match a ground-truth box. True positives divided by true positives plus false positives.
- **Recall** is the fraction of ground-truth boxes that were found. True positives divided by true positives plus false negatives.

A true positive is a prediction matched to a ground-truth box of the same class at or above the matching IoU. An unmatched prediction is a false positive. An unmatched ground-truth box is a false negative. The match is greedy by confidence, and a ground-truth box is used once.

### mAP@0.5 and mAP@0.5:0.95

Average precision (AP) summarizes the precision-recall curve for one class at one matching IoU. Mean average precision (mAP) is the unweighted mean of AP across classes.

- **mAP@0.5** uses a single matching IoU of 0.50. A box that roughly covers the object counts.
- **mAP@0.5:0.95** averages AP over matching IoU thresholds from 0.50 to 0.95 in steps of 0.05 (ten thresholds). This is the primary COCO detection metric. A loose box fails the stricter thresholds, so the number is lower than mAP@0.5 on the same predictions. COCO's detection evaluation is specified at [cocodataset.org/#detection-eval](https://cocodataset.org/#detection-eval).

Per-class AP is stored as well. A strong mean can hide a class the model never finds.

## Dataset and splits

### COCO

COCO is the reference dataset for the pretrained checkpoint. The dataset page is [https://cocodataset.org](https://cocodataset.org). The detection task uses 80 classes. Lin et al. (2014) describe the dataset.

Splits, used as COCO defined them:

- **train2017** is for training, if a later experiment trains on COCO. The first stage does not train. It loads the published checkpoint.
- **val2017** is the split the pretrained comparison is scored on. Ultralytics publishes its detection mAP against COCO validation. We do not sweep thresholds on val2017 and then present the swept score as that comparison. The comparison uses the standard validation settings.
- **test2017** labels are not public. This project will not claim a test2017 number.

Images are downloaded by a script into `data/` or read from a COCO root the config already points at. They are gitignored.

A smoke run on a handful of val2017 images is allowed while the script is being wired. A smoke run is not a COCO mAP. The reportable COCO number uses the full val2017 split. If that download is too large for the machine at hand, the README stays empty for COCO rather than quoting the subset.

### Custom dataset

The fine-tune needs a small set the COCO classes only partly cover. Helmet use is the preferred subject.

Candidates, checked when this plan was written:

- [motocyclehelmet_detection](https://universe.roboflow.com/deteksihelm/motocyclehelmet_detection) on Roboflow Universe, with the labels `pakai-helm` and `tidak-pakai-helm`.
- [Deteksi Helm Kendaraan Bermotor](https://universe.roboflow.com/abdul-jalil-azis/deteksi-helm-kendaraan-bermotor-pwech), with the labels `pakai helm` and `tanpa helm`.

The download step re-reads the class list from the version it actually fetches. Label names on those pages can change. The other path is to collect and label images for this project (Label Studio, CVAT, or Roboflow's annotator are all sufficient). A few hundred images are enough to show fine-tuning. The write-up will describe that size as a limit of the result.

Before any training run:

1. Confirm the license and record it in the config. If the license forbids training or redistribution, pick another set.
2. Freeze three splits: **train**, **validation**, and **test**.
3. If the source already publishes those splits, keep them. Do not move an image after looking at a score.
4. If we create the splits, stratify by the classes present in each image, use one fixed seed, and write the image ids to a file. A starting ratio is 70% train, 15% validation, 15% test. The ratio is a config value. The file of ids is hashed into every run.

Validation is where checkpoint selection, early stopping, and the demo's default confidence and IoU are decided. The test split is scored once, by `eval/evaluate.py`, after that config is fixed. A run whose config changed after the test score was seen is not reportable. `metrics.json` carries `"reportable": false` in that case, and the README does not quote it.

Class counts per split are printed before training. A split that dropped one of the classes is rejected.

## Model plan

Three stages, in order. A later stage starts after the previous one has a saved run, or after it has been explicitly skipped in this file.

### Stage 1 — Pretrained inference

Load one COCO-pretrained Ultralytics nano model and run it.

- Default checkpoint: YOLO11n (`yolo11n.pt`).
- Alternative, same size class, config change only: YOLOv8n (`yolov8n.pt`).
- Input size: 640, matching the tables Ultralytics publishes for these weights.
- Task: draw boxes, labels, and confidences on local images. No training.

The weights download on first use into the Ultralytics cache, outside git. The run record stores the filename, the Ultralytics version, and a hash of the weight file once it is on disk.

### Stage 2 — Fine-tune

Fine-tune the stage 1 checkpoint on the custom training split.

- Initialization: the pretrained nano weights, not a random network.
- Selection: the checkpoint with the best validation mAP@0.5:0.95. Epoch count, learning rate, freeze schedule, and augmentation live in `configs/finetune-helmets.yaml` (the name follows the dataset; a vehicle config gets its own file).
- Test: one evaluation after the config is fixed.
- The write-up separates COCO-class behavior of the untouched checkpoint from the custom classes of the fine-tuned one. A helmet class the pretrained model does not have is not a failure of the COCO run.

Overfitting is the expected risk on a few hundred images. The validation curve is part of the saved run. A widening gap between train and validation is reported with the score, not hidden behind the best epoch.

### Stage 3 — Optional Faster R-CNN baseline

If time allows, compare against torchvision's Faster R-CNN with a ResNet-50 FPN backbone and the COCO-pretrained weights torchvision documents for that model. The weight enum is pinned in `configs/fasterrcnn-coco.yaml` when the stage is built.

The comparison uses the same images and the same metric code. The write-up states the differences that remain: anchors, the input size that model expects, and its own NMS. Where the protocols differ, the numbers are shown side by side with those differences, and neither number is relabeled to match the other.

This stage does not block the demo. Skipping it leaves the README without a Faster R-CNN row.

### Published COCO reference

These figures are printed by Ultralytics. They are not DetectIt measurements. The evaluation write-up cites them and then places our own val2017 run next to the one that matches the checkpoint we loaded.

Read from the COCO detection performance tables on 2026-11-26:

| Checkpoint | Input | Published column | Published value | Source |
| --- | --- | --- | --- | --- |
| YOLOv8n | 640 px | mAP val 50-95 | 37.3 | [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/models/yolov8/) |
| YOLO11n | 640 px | mAP val 50-95 | 39.5 | [Ultralytics YOLO11 docs](https://docs.ultralytics.com/models/yolo11/) |

The column those pages label `mAP val 50-95` is the published mAP@0.5:0.95 for the COCO-pretrained detection weights. The same tables also list speed, parameter count, and FLOPs. Those columns are not copied into the README. Latency in this project is measured by our script.

If Ultralytics revises a table, the run cites the page as fetched for that run. The JSON stores the number we compared against and the URL. A number remembered from this plan, after the page has changed, is not the reference.

## Evaluation plan

### What is measured

On the declared split, for the declared checkpoint:

- **mAP@0.5**
- **mAP@0.5:0.95**
- **Precision** and **recall** at the operating confidence, with that confidence written next to them
- **Per-class AP** at the same matching setup as the headline mAP
- **Latency and throughput** at batch size 1: milliseconds per image, and frames per second computed from that mean. Measured on CPU. Measured on GPU when a GPU is present, with the device name stored. A machine with no GPU leaves the GPU fields out. The README does not fill them in.

Warmup iterations are discarded and the count of warmup and timed images is stored. The script computes frames per second. The README does not divide by hand.

### Pretrained comparison

For YOLO11n or YOLOv8n, the COCO comparison is:

- our mAP@0.5:0.95 on the full val2017 split, from `metrics.json`
- the published value in the table above for that same checkpoint, cited to the Ultralytics page

The text says which number is ours and which number is theirs. A gap is described. It is not treated as a defect to be closed by tuning on val2017.

### What a reportable run contains

Every run that might be quoted lives in `eval/runs/<run-id>/`:

- `config.yaml` — the resolved config, including weights, image size, confidence, NMS IoU, matching IoU, and seed
- `metrics.json` — the metrics listed above, plus model name, weights filename, weights hash, dataset name, split, split-file hash, and git revision
- `train.log` or `eval.log` — the console log of the script that wrote the JSON
- `environment.txt` — Python, PyTorch, torchvision, Ultralytics, and OpenCV versions

`metrics.json` sets `"reportable": true` only when the script finished the declared split from a clean git tree. The script lives in `eval/` and calls library code under `src/detectit/`. A hand-edited JSON file is not a result. A notebook cell is not a result.

### Evidence rule

Every number in [README.md](README.md) comes from a script in `eval/` that was actually run, with the log and the metrics JSON saved, and with the model, config, and dataset versions recorded in that JSON. A number that was not produced this way is not written into the README. That includes estimates, smoke runs, validation scores presented as test scores, unpublished leaderboard memories, and figures typed in by hand.

Two kinds of numbers are allowed to appear beside that rule, and both are labeled:

- **External published figures**, such as the Ultralytics COCO table above, with the URL. They are never presented as a DetectIt measurement.
- **The sibling line for VisionClassify.** The README states that project's reported CIFAR-10 test accuracy, 95.71%, because that is the description of the sibling. VisionClassify measured it. DetectIt did not. It is not evidence of anything in this repository.

Dataset facts that define a public dataset, such as COCO's 80 detection classes, are stated with the dataset link. They are not model scores.

## Streamlit app

The demo is a single app at `app/streamlit_app.py`. It calls the same preprocess, model, and NMS code as `eval/`.

### Inputs

- Upload an image (JPEG, PNG, or WebP).
- Upload a video file that OpenCV can decode. The page shows one frame at a time, with a frame slider.
- Optional webcam. A still frame from the browser camera is enough. A live stream is extra machinery and waits until image and video upload both work.
- Confidence threshold slider.
- IoU threshold slider, wired to NMS.
- Checkpoint selector. The app reads the config saved with that checkpoint and uses its input size and class names.

The sliders start at the defaults stored in the loaded config. Those defaults are the library's predict defaults until a sweep on the validation split replaces them. The test split is not part of that sweep.

### What the page shows

- The image or the current frame, with boxes, class labels, and confidences drawn in the original pixel space.
- A table of counts per class, counted from the boxes that survived both thresholds on the frame currently shown.
- The checkpoint name, the config name, and the dataset the weights were trained on.
- A short note when the count is zero.

For a video, an optional clip total sums the per-frame counts and states that an object visible for many frames is counted many times. Tracking is out of scope, so the total is not a count of unique objects.

### Before a checkpoint is wired

The page still accepts an upload and explains that predictions need a checkpoint and the installed package. It does not draw invented boxes.

## Repository structure

```text
DetectIt/
├── README.md                       # public summary; DetectIt scores only from eval runs
├── PLAN.md                         # this plan
├── docs/
│   ├── pipeline.svg                # static pipeline figure
│   └── pipeline.png
├── data/                           # COCO, custom images, weights (gitignored)
├── notebooks/                      # box-format checks, sample drawings
├── src/detectit/
│   ├── data/                       # download, splits, dataset, box conversion
│   ├── models/                     # Ultralytics wrapper, optional Faster R-CNN
│   ├── train/                      # fine-tuning loop
│   ├── eval/                       # metric helpers shared with the scripts
│   └── utils/                      # drawing, per-class counts, seeding
├── eval/                           # scripts that write logs and metrics JSON
│   ├── evaluate.py
│   ├── train.py
│   └── runs/                       # one directory per run (weights gitignored)
├── app/
│   └── streamlit_app.py
├── configs/                        # yolo11n-coco.yaml, yolov8n-coco.yaml, finetune-*.yaml
└── tests/                          # box conversion, IoU, NMS, splits, reportable flag
```

`src/detectit/eval/` holds functions. `eval/` at the repository root holds the scripts a person runs. The README may quote a DetectIt number only when one of those scripts wrote it.

`data/`, downloaded weights, and checkpoints inside `eval/runs/` are gitignored. Each quoted run still commits its `metrics.json`, log, resolved config, and environment file, so the README number has a file behind it.

## Milestones

A phase is finished when its tasks are done and, where the phase produces a measurement, the matching directory under `eval/runs/` exists and is the source of any README edit.

### Phase 0 — Plan and skeleton

- [x] Add `README.md`, `PLAN.md`, and the pipeline figure (`docs/pipeline.svg`, `docs/pipeline.png`).
- [ ] Add the package layout above, with `pyproject.toml` or an equivalent, and empty modules that import.
- [ ] Pin the planned runtime: Python, PyTorch, Ultralytics, OpenCV, Streamlit, torchvision.
- [ ] Add `.gitignore` for virtual environments, `data/`, weights, and run checkpoints.
- [ ] Add the MIT `LICENSE` file in the same commit that records the Ultralytics AGPL-3.0 decision.
- [ ] Confirm `pytest` collects, even if the first tests are the box-format round trip only.

### Phase 1 — Pretrained inference

- [ ] Config for YOLO11n at 640 pixels on the COCO class list.
- [ ] Config swap for YOLOv8n, same script.
- [ ] Load the weights, run them on a local image, and save a drawn image plus the raw predictions.
- [ ] Unit-test the box conversion used to draw that image.
- [ ] Leave the README results section empty. This phase produces a picture, not a COCO score.

### Phase 2 — Evaluation harness

- [ ] `eval/evaluate.py` writes `metrics.json`, the log, the resolved config, and `environment.txt`.
- [ ] Implement mAP@0.5, mAP@0.5:0.95, precision, recall, and per-class AP on absolute `xyxy` boxes.
- [ ] Implement batch-1 latency on CPU, and on GPU when one is present.
- [ ] Smoke the script on a few val2017 images. Keep `"reportable": false`.
- [ ] Run the loaded nano checkpoint on full COCO val2017.
- [ ] Write the comparison in the README from that JSON, next to the cited Ultralytics value for the same checkpoint.

### Phase 3 — Custom dataset

- [ ] Choose the helmet set or the self-labeled set, and record the license and version.
- [ ] Download or label into `data/`, gitignored.
- [ ] Write the train, validation, and test id file before any training, and test that the splits are disjoint and cover the images.
- [ ] Notebook: a handful of images with boxes drawn from the label files, one box format named in the title.

### Phase 4 — Fine-tune

- [ ] `configs/finetune-helmets.yaml` (or the vehicle equivalent) starting from the stage 1 weights.
- [ ] Training loop with validation mAP@0.5:0.95 for checkpoint selection.
- [ ] One test evaluation after the config is fixed. Archive the run.
- [ ] Update the README from that JSON, including the train-versus-validation gap.

### Phase 5 — Streamlit demo

- [ ] Image upload, boxes, labels, confidences.
- [ ] Video upload with a frame slider and the same drawing path.
- [ ] Confidence and IoU sliders wired to the shared post-processing.
- [ ] Per-class count table for the frame on screen.
- [ ] Checkpoint and config named on the page.
- [ ] Empty-checkpoint state that explains why there are no boxes.

### Phase 6 — Optional Faster R-CNN

- [ ] torchvision Faster R-CNN, COCO-pretrained, weight enum pinned in config.
- [ ] The same metric script on the custom test split, and on COCO val2017 if that run is affordable.
- [ ] README row sourced from that run's JSON, with the protocol differences stated.

### Phase 7 — Optional webcam

- [ ] Browser-camera still frame through the same predict path as an upload.
- [ ] A live stream only if the still frame is already solid and the extra dependency is worth it.

## Risks and mitigations

| Risk | What we will do |
| --- | --- |
| The test split influences a threshold or an epoch, and the score looks better than the protocol allows. | Validation is the only split used for choices. `eval/evaluate.py` scores the test split after the config is committed. A later config change marks the run not reportable. |
| YOLO normalized `cxcywh`, COCO `xywh`, and torchvision `xyxy` get mixed. | Conversion functions name both formats. A round-trip test uses a fixed box. Drawing and metrics share the absolute `xyxy` output. |
| Headline mAP is computed at the demo's confidence threshold, which cuts the precision-recall curve. | mAP uses the standard validation confidence. Precision and recall name the operating point. Both values are fields in `metrics.json`. |
| The NMS IoU and the matching IoU are stored as one number. | The config and the JSON have separate fields. The app slider changes NMS only. |
| A smoke subset of val2017 is quoted as COCO mAP. | `"reportable": true` requires the full declared split. The README links the run id. |
| Ultralytics' published mAP is pasted into the README as if we had measured it. | External figures stay cited to the Ultralytics page. Our number comes from `metrics.json`. The results paragraph says which is which. |
| The custom set is small and the fine-tune memorizes it. | The train and validation curves are saved. The README reports the gap with the test score. Augmentation is a config change with its own run, not a silent default. |
| Helmet boxes are tiny, occluded, or mislabeled, and mAP moves for reasons the model did not cause. | The phase 3 notebook shows real boxes before training. Label errors found then are fixed in a new dataset version, with a new split hash. |
| One class is missing from validation or test. | The split writer prints class counts and fails if a class count is zero. |
| A public dataset's license does not allow training or the images get committed under MIT. | The license is recorded before download. `data/` is gitignored. The README states that dataset licenses stay with the datasets. |
| Ultralytics is AGPL-3.0, and an MIT `LICENSE` file would misstate what a downstream user receives. | The license file is added in the commit that imports the package, and that commit states the AGPL obligation. Weights are not relicensed by this repository. |
| No GPU is available, and a GPU latency cell is filled with a guess. | GPU fields are written only by a run that used a GPU. Otherwise they are absent, and the README says CPU only. |
| Video frame counts are read as unique vehicles. | The table is per frame. A clip total, if shown, says that it counts each frame separately. |
| A score is typed into the README from memory. | The evidence rule above is the gate. Each DetectIt number in the README names its `eval/runs/` directory. |

## References

- Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. "Microsoft COCO: Common Objects in Context." *European Conference on Computer Vision*, 2014. Dataset: [https://cocodataset.org](https://cocodataset.org). Detection evaluation: [https://cocodataset.org/#detection-eval](https://cocodataset.org/#detection-eval). Annotation format: [https://cocodataset.org/#format-data](https://cocodataset.org/#format-data).
- Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks." *Advances in Neural Information Processing Systems*, 2015.
- Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. "You Only Look Once: Unified, Real-Time Object Detection." *IEEE Conference on Computer Vision and Pattern Recognition*, 2016. Background for the YOLO line. The checkpoints this project loads are the Ultralytics models below, not the 2016 network.
- Ultralytics. YOLOv8 documentation, including the anchor-free head description and the COCO detection performance table. [https://docs.ultralytics.com/models/yolov8/](https://docs.ultralytics.com/models/yolov8/). Table read 2026-11-26: YOLOv8n, 640 px, mAP val 50-95 = 37.3.
- Ultralytics. YOLO11 documentation, including the COCO detection performance table and the note on validation confidence versus an operating point. [https://docs.ultralytics.com/models/yolo11/](https://docs.ultralytics.com/models/yolo11/). Table read 2026-11-26: YOLO11n, 640 px, mAP val 50-95 = 39.5.
- Ultralytics. Detection datasets and the normalized YOLO label format. [https://docs.ultralytics.com/datasets/detect/](https://docs.ultralytics.com/datasets/detect/).
- torchvision detection models, for the optional Faster R-CNN baseline. [https://pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html).
- Roboflow Universe, candidate helmet sets: [motocyclehelmet_detection](https://universe.roboflow.com/deteksihelm/motocyclehelmet_detection), [Deteksi Helm Kendaraan Bermotor](https://universe.roboflow.com/abdul-jalil-azis/deteksi-helm-kendaraan-bermotor-pwech).
- Ultralytics package license: GNU Affero General Public License v3. [https://github.com/ultralytics/ultralytics/blob/main/LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
