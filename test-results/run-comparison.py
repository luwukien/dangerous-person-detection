from ultralytics import YOLO
from pathlib import Path
import csv
root = Path.cwd()
rows=[]
for variant in 'nsml':
    model = YOLO(str(root / 'model-comparison' / f'yolo11{variant}-fit-within-640' / 'weights' / 'best.pt'))
    result = model.val(data=str(root / 'datasets/data-local-eval.yaml'), split='test', imgsz=640, batch=8, device='cpu', workers=0, project=str(root / 'test-results'), name=f'yolo11{variant}', exist_ok=True, plots=True)
    rows.append([f'YOLO11{variant}', result.box.mp, result.box.mr, result.box.map50, result.box.map])
    with (root / 'docs/images/test-comparison.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['Model','Precision','Recall','mAP50','mAP50-95']); w.writerows(rows)
