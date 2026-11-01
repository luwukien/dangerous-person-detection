# Dangerous Person Detection

Person-level object detection using Ultralytics YOLO11. The current experiments compare YOLO11n, YOLO11s, YOLO11m and YOLO11l on a two-class dataset, with locally available training logs, checkpoints, image inference code and an experimental video-review application.

> **Project status:** research prototype. The validation results below describe this project's supplied comparison, not the original upstream study. The local application has not been evaluated as a production security system.

## Detection task

The model draws bounding boxes around **people**, rather than around weapons alone.

| ID | Class | Annotation meaning |
| ---: | --- | --- |
| 0 | `normal_person` | A reviewed person without clear visual evidence of the class 1 object association. |
| 1 | `potentially_dangerous_person` | A reviewed person visibly holding, carrying or operating a gun or knife under the annotation rules. |

Class 1 describes visible image context. It does not establish identity, intent, guilt or a real-world threat level. Predictions require human review.

## Dataset

The local `datasets/` directory contains **3,639 images** and **11,822 labeled person instances**. These counts were checked directly against image and YOLO label files and agree with the saved image counts in the [dataset preparation notebook](notebooks/prepare_split_dataset_640_640_fitwithin.ipynb).

| Split | Images | Class 0 instances | Class 1 instances | Total instances | Images with empty labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 2,911 | 6,743 | 2,684 | 9,427 | 55 |
| Validation | 364 | 840 | 329 | 1,169 | 10 |
| Test | 364 | 917 | 309 | 1,226 | 6 |
| **Total** | **3,639** | **8,500** | **3,322** | **11,822** | **71** |

There is one label file per image, with no missing label files in the inspected copy. An empty label file contains no annotated boxes; this alone does not prove that the image was manually reviewed. Image counts and person-instance counts are different quantities.

### Sources and preparation

The notebook downloads these Roboflow exports from `chikiens-workspace` in YOLO format:

| Roboflow project | Version | Train images | Validation images | Test images |
| --- | ---: | ---: | ---: | ---: |
| `1-prj-person-detection-dataset` | 2 | 800 | 100 | 100 |
| `2-prj-person-detection-dataset` | 1 | 799 | 100 | 100 |
| `3-prj-person-detection-dataset` | 1 | 800 | 100 | 100 |
| `4-prj-person-detection-dataset` | 1 | 512 | 64 | 64 |

It merges the existing `train`, `valid` and `test` directories, prefixes filenames with `ds1_` through `ds4_` to prevent overwriting, and writes a combined `data.yaml`. The resulting proportions are approximately 80/10/10. Despite its filename, the notebook does not implement resizing; the verified model input setting is `imgsz=640` during training and inference.

**Split limitation:** preserving source splits and adding filename prefixes does not establish scene independence or remove duplicates. Manual annotation review, source provenance checks, and checks for related frames or duplicate content across splits remain necessary. Keep related images in the same reviewed group and freeze the test set before selecting models or tuning thresholds.

The [dataset pipeline guide](docs/DATA_PIPELINE.md) retains annotation, provenance and grouped-splitting guidance. Its downloader and preparation commands refer to earlier tooling absent from the current working copy. Use the notebook and scripts described here for the present workflow. The URLs in [config/datasets.json](config/datasets.json) are collection references, not proof that each source contributed to this merged dataset.

## Model comparison

These **validation** metrics come from the locally supplied `model-comparison/comparison.csv`. The experiment owner confirmed the evaluation split; the CSV itself does not record the split, evaluation command, confidence threshold or hardware. These are reported artifacts, not a new evaluation performed for this documentation update.

| Model | Precision | Recall | mAP50 | mAP50-95 | Inference (ms/image) | Weights (MB, CSV) | Training time (h) | Recorded epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n | 0.8628 | 0.8907 | 0.9312 | 0.8101 | 3.97 | 5.25 | 2.036 | 141 |
| YOLO11s | 0.8759 | 0.8731 | **0.9444** | 0.8213 | 6.82 | 18.28 | 1.6667 | 116 |
| YOLO11m | 0.8398 | **0.9065** | 0.9355 | **0.8219** | 17.78 | 38.64 | 3.04 | 133 |
| YOLO11l | **0.8813** | 0.8855 | 0.9316 | 0.8188 | 22.97 | 48.82 | 4.8 | 150 |

- **YOLO11s** has the highest reported mAP50 and is the default variant referenced by the inference code.
- **YOLO11m** has the highest reported recall and mAP50-95; **YOLO11l** has the highest precision.
- **YOLO11n** has the lowest reported inference time and smallest checkpoint.

Training times are **owner-reported decimal hours**: YOLO11m 3.04 h, YOLO11s 1 h 40 min (approximately 1.6667 h), YOLO11l 4 h 48 min (4.8 h), and YOLO11n 2.036 h. They are separate from the cumulative `time` column in training logs. Recorded epochs come from each run's `results.csv`, not the configured maximum.

The CSV's weight sizes correspond to checkpoint bytes divided by 1,048,576 (MiB), although the column is labeled MB. Hardware and benchmark details are unavailable, so the inference times do not establish expected CPU, video or website performance. No held-out test metrics are reported here.

## Training configuration

The local entry point is `scripts/train.py`. Its settings agree with the saved `args.yaml` files under `model-comparison/yolo11*-fit-within-640/`:

| Setting | Value |
| --- | --- |
| Task | Two-class person detection |
| Initialization | Pretrained `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt` |
| Input size / batch size | `imgsz=640` / 8 |
| Maximum epochs | 150 |
| Early-stopping patience | 35 for n/s; 40 for m/l |
| Optimizer | `auto` |
| Seed / deterministic | 42 / `True` |
| Mixed precision / data workers | AMP enabled / 2 |
| Horizontal / vertical flip | 0.5 / 0.0 |
| Mosaic / MixUp | 1.0 / 0.0 |
| Close mosaic | Final 10 epochs |
| Checkpoint interval | Every 10 epochs, plus best checkpoint |
| Recorded training device | CUDA device `0`; GPU model not recorded in the inspected artifacts |

The actual optimizer selected by `auto` and the exact training environment are not established by these settings. The local video dependency file pins `ultralytics==8.4.197`; this does not establish the version used for all training runs.

### Run training

The following commands assume the local training script and dataset have been supplied; a fresh Git clone may not include them. Create and activate a Python environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install ultralytics
```

Update `datasets/data.yaml` so `path` points to your absolute local `datasets` directory. Keep `train: train/images`, `val: valid/images`, `test: test/images` and the class order above. The inspected YAML still points to `/content/drive/MyDrive/datasets`, so it requires this change before local use.

```powershell
python scripts/train.py --variant yolo11s --data datasets/data.yaml --device 0 --project runs/train
```

Use `--variant yolo11n`, `yolo11m` or `yolo11l` for other models, or `--device cpu` for CPU training. The default run name is `<variant>-fit-within-640`; existing run directories are not overwritten.

For Kaggle, point the dataset YAML at the mounted dataset and pass it with `--data`. Recorded runs used `/kaggle/working/data_kaggle.yaml` and wrote outputs under `/kaggle/working/model-comparison`. Preserve the resolved configuration, environment, checkpoint and evaluation command with future results.

## Image inference

The [image inference script](scripts/inference.py) accepts an image, optional `--weights`, and optional `--output`. It uses `imgsz=640`, confidence threshold **0.25**, and CUDA device 0 when available, otherwise CPU. It saves an annotated image and prints classes, confidence scores and box coordinates.

```powershell
python scripts/inference.py path/to/image.jpg --weights model-comparison/yolo11s-fit-within-640/weights/best.pt --output inference-results
```

Pass `--weights` explicitly: the script's default still includes the old `kaggle-output/` prefix, while the inspected artifacts are under `model-comparison/`. Results are saved under a new `yolo11s-single-image` run directory in the selected output location.

## Local video demo

The local `video_app/` application accepts MP4, MOV, AVI and MKV uploads, displays progress, and produces a downloadable MP4 with person boxes, labels and confidence scores. It processes frames on **CPU**, one video at a time; output has no audio.

With the local application files and checkpoint available, run:

```powershell
python -m pip install -r video_app/requirements.txt
$env:MODEL_PATH = (Resolve-Path "model-comparison/yolo11s-fit-within-640/weights/best.pt").Path
python -m uvicorn video_app.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Open `http://127.0.0.1:8000`. `MODEL_PATH` overrides the old checkpoint path. Optional positive integer environment variables `MAX_VIDEO_MB` and `MAX_VIDEO_SECONDS` limit upload size and duration; both limits are disabled by default. Supported decoding is limited to 3840 x 2160 and 120 FPS.

This single-user demo has no authentication; keep it bound to localhost. Working files are temporary and removed on normal shutdown; download results before stopping it. Abrupt termination may leave temporary files. The local `docs/VIDEO_DEMO.md` has additional usage details, but its default checkpoint path also retains the old prefix. No Azure deployment is claimed here.

## Files and artifacts

| Path | Purpose |
| --- | --- |
| `notebooks/prepare_split_dataset_640_640_fitwithin.ipynb` | Roboflow downloads, split-preserving merge and saved counts |
| `datasets/{train,valid,test}/{images,labels}` | Local images and YOLO labels |
| `datasets/data.yaml` | Local dataset paths and class mapping |
| `scripts/train.py` | Local YOLO11 training entry point |
| `scripts/inference.py` | Single-image inference CLI |
| `model-comparison/comparison.csv` | Supplied validation comparison |
| `model-comparison/yolo11*-fit-within-640/` | Local arguments, logs, plots and checkpoints |
| `video_app/` | Local FastAPI video demo and tests |
| `docs/DATA_PIPELINE.md` | Annotation and provenance guidance; earlier tool references |
| `docs/AGENT_CONTEXT.md` | Archived upstream research context |

These paths describe the inspected working copy, including local files that may not be committed or distributed with Git. The README update alone does not distribute datasets, checkpoints or the local application. Obtain required artifacts separately before running their commands. Non-empty `best.pt` files were found for all four variants; some intermediate epoch checkpoints are empty, so filenames alone do not establish usable weights.

## Research lineage and limitations

This work takes its person-level task from [Tran Si Nam's YOLO11 research prototype](https://github.com/Nambekai/dangerous-person-detection-yolo11). The [archived context](docs/AGENT_CONTEXT.md) preserves that study's description. Its dataset and test metrics are historical upstream results and must not be substituted for the dataset counts and validation comparison above.

Manual review of annotations, source permissions, related scenes, false positives and missed people remains essential. The available merge code does not verify independence between splits, and the supplied comparison lacks a complete evaluation record. Keep the test split separate from model selection and threshold tuning. No production security reliability or deployment performance is established by these experiments.

The repository does not declare a license for its own code or data. Review dataset terms, checkpoint provenance and applicable software licenses before reuse or redistribution. Keep API keys and credentials outside committed files.
