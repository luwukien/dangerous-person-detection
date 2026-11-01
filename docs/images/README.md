# README figure sources

These assets illustrate the current project's local experiment artifacts. They do not reuse the reference project's prediction images or historical metrics.

| Assets | Source |
| --- | --- |
| `project-banner.png`, `system-workflow.png` | Project overview diagrams generated with Matplotlib; the workflow is a schematic, not a model output |
| `dataset-overview.png`, `dataset-counts.json` | Counts read from local `datasets/{train,valid,test}/{images,labels}` |
| `model-comparison.png`, `model-cost.png`, `model-comparison.csv` | Local `model-comparison/comparison.csv`; copied CSV retains its original header and precision |
| `training-history.png` | Per-epoch validation mAP from the four local `model-comparison/yolo11*-fit-within-640/results.csv` files |
| `yolo11s-*.png`, `yolo11s-*.jpg` | Unmodified plot and validation-batch files from `model-comparison/yolo11s-fit-within-640/` |
| `hugging-face-demo.png` | Screenshot supplied by the project owner |

`sources.json` records original repository-relative paths and SHA-256 checksums for the eight copied YOLO11s outputs. These images preserve all original labels, scores and plotting-tool formatting. The exact evaluation command/checkpoint for the saved diagnostic plots is not established by the images alone. They are distinct from the owner-confirmed validation comparison CSV.

To regenerate with the original local datasets and all four run directories available:

```powershell
python -m pip install matplotlib
python scripts/build_readme_assets.py
```

Run from the repository root. This only counts labels, plots saved numbers and copies existing images; it does not train or run inference. The script refreshes its named assets and data snapshots, and leaves the supplied Hugging Face screenshot unchanged. The README retains the source-provenance and dataset-reuse limitations of the project.
