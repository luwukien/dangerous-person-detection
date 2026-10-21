"""Download independent Kaggle sources without changing their annotations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


def dataset_handle(url):
    parsed = urlparse(url)
    parts = parsed.path.strip('/').split('/')
    if (parsed.scheme != 'https' or parsed.hostname not in {'kaggle.com', 'www.kaggle.com'}
            or len(parts) != 3 or parts[0] != 'datasets'
            or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', p) for p in parts[1:])):
        raise ValueError(f'Invalid Kaggle dataset URL: {url}')
    return '/'.join(parts[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/datasets.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    urls = json.loads(args.config.read_text(encoding='utf-8-sig'))['urls']
    handles = {dataset_handle(url): url for url in urls}
    import kagglehub

    def download(handle):
        destination = args.output.resolve() / handle
        if destination.exists():
            raise ValueError(f'Destination exists; choose a new --output: {destination}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        return kagglehub.dataset_download(handle, output_dir=str(destination))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(download, h): h for h in handles}
        for task in as_completed(tasks):
            handle = tasks[task]
            row = {'handle': handle, 'url': handles[handle],
                   'requested_version': 'latest', 'resolved_version': None,
                   'downloaded_at': datetime.now(timezone.utc).isoformat()}
            try:
                row.update(status='ok', path=str(task.result()))
                print(f'[OK] {handle}: {row["path"]}')
            except Exception as error:
                row.update(status='error', error=str(error))
                print(f'[ERROR] {handle}: {error}')
            results.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    (args.output / f'download-{stamp}.json').write_text(
        json.dumps(results, indent=2), encoding='utf-8')
    return int(any(r['status'] == 'error' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
