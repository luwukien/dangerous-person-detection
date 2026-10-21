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
   during splitting. Do not delete rows. Similarity uses a 64-bit perceptual hash
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
