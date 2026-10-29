# Manual person dataset pipeline

This is a new dataset preparation pipeline, not a reproduction of the historical
3,140-image dataset. The prediction unit is a person. Class IDs are fixed:
`0 normal_person`, `1 potentially_dangerous_person`.

## Install and download

Run commands from the repository root with Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_datasets.py
```

Add URLs to `config/datasets.json`. Downloads run two at a time; failures are
reported independently and the process returns a nonzero exit code if any fail.
Existing destination directories are never overwritten. For retries, put failed
URLs in a separate config and use a fresh output directory, then prepare that
directory with `--raw`. Download reports record timestamps and URLs; the resolved
Kaggle version is explicitly unknown rather than inferred. Preserve source license
notices and record the version from Kaggle before publishing the dataset.

If Kaggle requires authentication, run `python -c "import kagglehub; kagglehub.login()"`.
Do not put credentials in the repository. Extract nested image archives manually
before preparation; original XML/JSON/TXT annotations are inventoried, not imported.

## Prepare and manually annotate

### Roboflow: prepare all unique images before splitting

Use this workflow to annotate on Roboflow and split locally afterward:

```powershell
.\.venv\Scripts\python.exe scripts/roboflow_dataset.py prepare
```

This reads loose images and image members in ZIP files under `data/raw`, without
modifying or extracting into the raw directory. Unsafe archive paths, damaged
images/archives, unsupported files, and incomplete downloads are reported in
`data/roboflow/inventory.json`. Videos are not sampled; ZIPs nested inside ZIPs
are reported as unsupported members. Original weapon labels are inventoried only.

The new `data/roboflow/` directory contains:

- `images/`: EXIF-oriented, RGB, lossless PNGs named by pixel hash. Exact duplicates
  across sources and archives share one image, with every origin in `manifest.json`.
- `upload/`: images-only ZIPs. The first batch has up to 12 images for a pilot;
  subsequent batches have up to 250 images. Each image belongs to exactly one batch.
- `upload-manifest.json`: batch image IDs, SHA-256 checksums, and blocked images.
  PNGs above 20 MB or 16400 x 10900 pixels stay locally and are explicitly blocked
  from upload; there is no automatic resize. Resolve these before claiming all
  images were uploaded. [Roboflow upload documentation](https://docs.roboflow.com/datasets/create-and-upload/adding-data.md).
- `source_provenance.json`: copies and hashes of source download manifests and
  available README/license notices. Missing source license information is not inferred.
- `review.csv`, `near_duplicates.csv`, `group_suggestions.csv`, and `classes.txt`:
  manual review records. Group suggestions derived from frame names require confirmation;
  they do not populate approved groups automatically.

The command refuses an existing work directory. Use `--work data/roboflow-v2`
for another version; pass that same work path to import, validate, and export.

#### Select up to 5,000 existing images for upload

If `data/roboflow/images` already exists, select a smaller images-only upload
without deleting or rewriting those PNGs:

```powershell
.\.venv\Scripts\python.exe scripts/select_roboflow_upload.py --preview
.\.venv\Scripts\python.exe scripts/select_roboflow_upload.py
```

The default target is the lesser of 5,000 and half of the eligible PNGs, using
seed 42. Oversize or unreadable PNGs are excluded. The script uses an existing
`manifest.json` when available; otherwise it scans `data/raw` and matches the
prepared pixel hashes to recover source paths. Known frame sequences stay together,
and sources are sampled in roughly their original proportions. Frame-name groups
are suggestions only, not confirmed evaluation groups. Images without recoverable
origins are marked `unknown`; their scene relationships still need manual review.
No classes are inferred from unannotated images.

`data/roboflow/upload-selected/selection-manifest.json` lists every selected ID,
source, group hint, blocked image, and batch. Its `upload_status` entries start as
`pending`; they do not prove that Roboflow received an image. Upload the ZIPs in
that folder to your Roboflow Object Detection project and check Roboflow's counts.
The script refuses to replace an existing selection directory, keeping the first
selection reproducible. Use a different `--output` path for a new selection.

#### Upload and annotate

1. Create an **Object Detection** project. Upload only `upload/batch-0001.zip`
   initially. Use the classes `normal_person` and `potentially_dangerous_person`.
   Draw boxes around every person according to the policy below, not around weapons.
2. If Roboflow requires a split, put all images into `train` temporarily. This
   assignment is ignored when creating the final local split.
3. Annotate the pilot. Export **COCO JSON**, disabling preprocessing (including
   resize/crop) and augmentation. Keep the prepared image geometry unchanged.
   [Roboflow COCO support](https://roboflow.com/formats/coco-json).
4. Extract the export to a separate folder, then import it:

```powershell
.\.venv\Scripts\python.exe scripts/roboflow_dataset.py import --export data/roboflow-export-pilot
```

Inspect the box previews under the `imports/<import-id>/previews/` path printed by
the command. Confirm the boxes, classes and orientation match before uploading
the remaining batches. A real Roboflow pilot requires your project and annotations;
offline tests do not replace this check.

After annotating the remaining images, export again with the same settings and
run import with the new extracted folder. The importer reads COCO JSON across all
export splits, maps class IDs by class name into the required local 0/1 order,
and converts COCO pixel boxes to normalized YOLO boxes. Unused parent categories
are allowed, but annotated unknown classes and crowd boxes are rejected.

Original hash filenames and Roboflow suffixes such as `_png.rf.<token>.jpg` are
matched back to the local manifest. Unknown names, changed dimensions, invalid
boxes, and conflicting duplicate labels block import before labels are written.
Export preprocessing that preserves dimensions cannot be detected from COCO
metadata alone: disabling transformations and checking pilot previews is required.
The local prepared PNGs remain the canonical images. Exported JSON documents and
their hashes are saved under `imports/` for provenance.

An identical import can be repeated. Existing differing labels are never replaced;
for revisions, make a new copy of the work directory, remove only the affected
label files from that copy, reset those review rows to `pending`, then import and
review again. Do not change the original work version.

#### Review, then split locally

Import leaves `review.csv` unchanged. An image present in COCO with no boxes gets
an empty label file but still requires explicit confirmation that no people are
visible. Images absent from the export are listed in `import-report.json`; missing
labels are never treated as negatives. The report describes the latest import;
earlier import reports remain under `imports/`.

Set `approved` only after reviewing every person/box, or `excluded` with a reason.
Every approved image in this workflow must also have a manually reviewed `group`:
use the same globally unique video/scene group across related frames and sources.
If independence cannot be established, keep images pending or conservatively group
the uncertain source together. A confirmed independent still image can have its
own group. Neither old source split folders nor Unity `split-*` ZIP names establish
independence. Review near-duplicate pairs; `pending` and `same` pairs stay together.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dataset.py validate --work data/roboflow
.\.venv\Scripts\python.exe scripts/prepare_dataset.py export --work data/roboflow
```

This produces `data/dataset/{train,valid,test}` with an 80/10/10 target and seed 42.
At least three independent groups are required. Actual ratios can differ because
groups cannot be split. Inspect the export report and previews; freeze the test
set before any future training. No split or training runs during preparation/import.

### Other annotation tools

```powershell
python scripts/prepare_dataset.py prepare
```

The preparation step writes `data/annotation/` and refuses an existing work
directory. Exact decoded RGB duplicates are collapsed, retaining every source in
`manifest.json`. Images are normalized for EXIF orientation and saved losslessly
as PNG with unique content-based names. Raw files are unchanged.

1. Open `data/annotation/images` in your annotation tool using `classes.txt` in
   exactly its supplied order. Export YOLO detection labels to
   `data/annotation/labels/<image-id>.txt`, preserving filenames.
2. Draw boxes tightly around the visible extent of each distinguishable person.
   Label all people, not just the person with a weapon. Class 1 requires clear
   visual evidence that this person is holding, carrying, or operating a gun or
   knife. A nearby unassociated weapon is insufficient. Class 0 is the remaining
   visually assessed context, not a claim about a person's character or intent.
3. People carrying umbrellas receive class 0 unless there is clear weapon evidence.
   If the object/person association or an essential annotation is ambiguous, mark
   the entire image `excluded` with a reason, or leave it `pending` for review.
4. In `review.csv`, set `status` to `approved` only after checking every person and
   every box. A verified image with no people needs an explicitly empty label file.
   A missing file is always an error for an approved image.
5. In the `group` column, assign the same globally unique video/sequence identifier
   to all its frames, including frames found in other sources. Use manifest paths
   to trace provenance. Leave unknown origins blank and document uncertainty in
   `notes`; filenames alone are not reliable evidence of independent scenes.
6. Review `near_duplicates.csv`: set `decision` to `same` for related images or
   `different` for false matches. `pending` pairs are conservatively kept together
   during splitting. Do not delete rows. Similarity uses a 64-bit difference hash
   at distance <=4 by default; it is a review aid and cannot find all duplicates.

Do not convert weapon boxes into person boxes. Original weapon annotations are
reference material only. If a source has no usable people, exclude those images.
Do not edit prepared PNGs after annotation; validation checks pixel hashes.

## Validate and export

```powershell
python scripts/prepare_dataset.py validate
python scripts/prepare_dataset.py export
```

Validation writes `data/annotation/validation.json`. Pending/excluded images are
reported and omitted; invalid approved images block export. At least three
independent groups are required. Export uses seed 42, seeks 80/10/10, and balances
image count, source, class instance counts and empty images with a deterministic
greedy assignment. Groups are indivisible, so exact ratios and class presence in
every split are not guaranteed. Inspect actual counts in `report.json`.

Output:

```text
data/dataset/
  train/{images,labels,previews}/
  valid/{images,labels,previews}/
  test/{images,labels,previews}/
  manifest.json
  report.json
  review.csv
  near_duplicates.csv
  inventory.json
config/data.yaml
```

Inspect previews from each source/split before training. These are spot checks,
not substitutes for reviewing all annotations. The export manifest records source
paths, reviewed labels, label hashes, groups and splits. Preserve it with the
download reports and environment versions. Existing outputs/YAML are never replaced;
use both `--output data/dataset-v2 --yaml config/data-v2.yaml` for a new version.
An interrupted export is incomplete; select new output paths for a retry.

Freeze the test set; do not use it to choose models or tune confidence thresholds.
Unknown video provenance and undetected near duplicates remain explicit limitations.
The historical metrics in the [agent context](AGENT_CONTEXT.md) do not apply to this new dataset.

Optional Ultralytics reader smoke check (install Ultralytics separately for training):

```powershell
python -c "from ultralytics.data.utils import check_det_dataset; print(check_det_dataset('config/data.yaml'))"
```

Run the offline pipeline tests:

```powershell
python -m unittest discover -s tests -v
```
