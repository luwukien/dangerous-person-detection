# Dangerous Person Detection

An auditable dataset preparation pipeline for person detection in images containing potentially dangerous objects. This repository collects image sources, supports manual review of **person-level** annotations, and exports grouped YOLO detection splits for future model training.

> **Project status:** dataset preparation. This repository does not contain a trained checkpoint or a training/inference application. The results reported by the [original YOLO11 research project](https://github.com/Nambekai/dangerous-person-detection-yolo11) have not been reproduced here.

## What this project does

| Stage | Current implementation |
| --- | --- |
| Source collection | Download configured Kaggle datasets into `data/raw/` and record download outcomes. |
| Image preparation | Normalize image orientation, preserve original files, collapse exact duplicates, and inventory source provenance; the Roboflow workflow also reads ZIP archives. |
| Human annotation | Prepare images-only Roboflow upload batches, import COCO boxes into YOLO, and review every visible person with explicit approval, exclusion, and grouping records. |
| Quality checks | Validate labels, image integrity, review status, and possible near-duplicate relationships before export. |
| Dataset export | Create deterministic, group-aware `train` / `valid` / `test` splits, preview images, reports, and `config/data.yaml`. |

The prediction unit is a **person**, with two contextual classes:

| ID | Class | Meaning |
| ---: | --- | --- |
| 0 | `normal_person` | A reviewed person without clear visual evidence of the class 1 object association. |
| 1 | `potentially_dangerous_person` | A reviewed person visibly holding, carrying, or operating a gun or knife under this dataset's annotation rules. |

Class 1 describes visible image context. It does not establish anyone's identity, intent, guilt, or real-world threat level. Outputs from any future model should be reviewed by a human.

## Quick start

Requirements: Python 3.10 or newer, access to the configured Kaggle datasets, and an annotation tool that exports YOLO detection labels.

```powershell
git clone https://github.com/luwukien/dangerous-person-detection.git
cd dangerous-person-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_datasets.py
python scripts/prepare_dataset.py prepare
```

The default source URLs are in [`config/datasets.json`](config/datasets.json). Kaggle may require authentication; keep credentials outside the repository. Extract image archives nested inside downloaded datasets before running `prepare`.

Preparation creates `data/annotation/`. Annotate its images with the class order in `classes.txt`, save YOLO labels under `labels/`, complete `review.csv`, and review `near_duplicates.csv`. An approved image with no people still needs an empty label file. Related frames and images must share a group so they cannot cross evaluation splits.

```powershell
python scripts/prepare_dataset.py validate
python scripts/prepare_dataset.py export
python -m unittest discover -s tests -v
```

Export produces `data/dataset/` and `config/data.yaml`. Inspect `report.json` and the split previews before using the data. Existing work and export directories are protected against overwrite; choose new paths for a new dataset version.

For the annotation policy, review fields, provenance rules, retry options, and output details, follow the [dataset pipeline guide](docs/DATA_PIPELINE.md).

### Annotate on Roboflow before splitting

```powershell
python scripts/roboflow_dataset.py prepare
# Upload data/roboflow/upload/batch-0001.zip first for a small pilot.
# Annotate on Roboflow, export COCO JSON without preprocessing/augmentation,
# and extract the export into data/roboflow-export-pilot.
python scripts/roboflow_dataset.py import --export data/roboflow-export-pilot
```

Inspect the imported box previews before uploading the remaining batches. This
workflow reads loose images and ZIPs from `data/raw`, preserves all sources, and
creates `data/roboflow/images` plus upload ZIPs without splitting the dataset.
To upload at most 5,000 images (and no more than half of eligible images) from an
existing `data/roboflow/images` directory, use the subset command below. It creates
new images-only ZIPs under `data/roboflow/upload-selected` and leaves the originals
untouched. Upload those ZIPs manually to your Roboflow Object Detection project.

```powershell
python scripts/select_roboflow_upload.py --preview
python scripts/select_roboflow_upload.py
```

The selection manifest records the seed, image IDs, source matches, group hints,
blocked files, and ZIP contents. See the dataset pipeline guide for limitations.
After annotating and importing all desired images, complete `data/roboflow/review.csv`
and review near-duplicate pairs. Every approved image requires a reviewed group.

```powershell
python scripts/prepare_dataset.py validate --work data/roboflow
python scripts/prepare_dataset.py export --work data/roboflow
```

See the [Roboflow workflow](docs/DATA_PIPELINE.md#roboflow-prepare-all-unique-images-before-splitting)
for upload limits, class mapping, negative images, and label revisions.

### Compare Roboflow versions on Colab

The [comparison notebook](notebooks/compare_roboflow_versions.ipynb) trains
YOLO11s independently on Roboflow versions 2 and 3, checks their exports and
compares validation metrics. Open it in Google Colab with a GPU and save a copy
to Drive. It asks for a Roboflow API key privately and writes results to Drive.
Manual annotation, source, preprocessing and group-boundary review is required
before training; the filename audit alone cannot establish dataset equivalence.
The held-out test split is not used for model selection.

## Repository structure

```text
config/datasets.json          Kaggle source URLs
download_datasets.py          Source downloader and download report
scripts/prepare_dataset.py   Prepare, validate, and export commands
scripts/roboflow_dataset.py  Prepare upload batches and import Roboflow COCO labels
docs/DATA_PIPELINE.md         Complete annotation and export guide
docs/AGENT_CONTEXT.md         Archived project and research context for agents
AGENTS.md                     Instructions for agents working in this repository
tests/                       Offline pipeline tests
```

Generated data and local environments are excluded from Git. Downloaded sources and annotations require their own provenance and license review before redistribution.

## Research lineage and limitations

This work takes its **person-level detection task** from [Tran Si Nam's YOLO11 research prototype](https://github.com/Nambekai/dangerous-person-detection-yolo11). The original project reports a 3,140-image dataset and YOLO11s test results, including mAP50 of 0.80. Those figures describe the original study only; this repository's new dataset has no published model results. The archived [agent context](docs/AGENT_CONTEXT.md) retains the earlier project description for development reference.

Manual annotation, source permissions, scene grouping, and review of false positives and missed people remain essential. Similarity checks cannot detect every related image, and group-preserving splits may differ from the requested 80/10/10 proportions. Keep the test split frozen when training or tuning a future model.

This repository currently does not declare a license for its own code or data. Check the [original project's license](https://github.com/Nambekai/dangerous-person-detection-yolo11/blob/main/LICENSE) and each source dataset's terms before reuse or redistribution.
