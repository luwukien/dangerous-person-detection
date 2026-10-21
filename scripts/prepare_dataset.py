"""Prepare manual person annotations, then validate and export grouped YOLO splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import imagehash
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['normal_person', 'potentially_dangerous_person']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pixel_hash(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def prepare(raw, work, distance):
    if not raw.is_dir():
        raise ValueError(f'Raw directory not found: {raw}')
    paths = sorted(p for p in raw.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('No images found; extract any nested archives first.')
    work.mkdir(parents=True, exist_ok=False)
    (work / 'images').mkdir()
    (work / 'labels').mkdir()
    by_hash, errors, inventory = {}, [], Counter()
    for path in paths:
        relative = path.relative_to(raw).as_posix()
        source = '/'.join(path.relative_to(raw).parts[:2])
        inventory[source] += 1
        try:
            with Image.open(path) as opened:
                im = ImageOps.exif_transpose(opened).convert('RGB')
                digest = pixel_hash(im)
                origin = {'source': source, 'path': relative}
                if digest in by_hash:
                    by_hash[digest]['origins'].append(origin)
                    continue
                # PNG preserves decoded pixels and fixes EXIF orientation for annotation.
                im.save(work / 'images' / f'{digest}.png')
                by_hash[digest] = {'id': digest, 'pixel_hash': digest,
                                   'phash': str(imagehash.phash(im)),
                                   'width': im.width, 'height': im.height, 'origins': [origin]}
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({'path': relative, 'reason': str(error)})
    records = list(by_hash.values())
    if not records:
        raise ValueError(f'No readable images; inspect {work}')
    write_json(work / 'manifest.json', records)
    write_csv(work / 'review.csv', ['id', 'status', 'group', 'notes'],
              [{'id': r['id'], 'status': 'pending', 'group': '', 'notes': ''} for r in records])
    # BK tree avoids comparing every pair when perceptual hashes are well distributed.
    tree, candidates = None, []
    for record in records:
        value = int(record['phash'], 16)
        stack = [tree] if tree else []
        while stack:
            node = stack.pop()
            d = (value ^ node[0]).bit_count()
            if d <= distance:
                candidates.append({'left': node[1], 'right': record['id'],
                                   'distance': d, 'decision': 'pending'})
            stack.extend(child for edge, child in node[2].items() if d-distance <= edge <= d+distance)
        new = [value, record['id'], {}]
        if tree is None:
            tree = new
        else:
            node = tree
            while True:
                d = (value ^ node[0]).bit_count()
                if d not in node[2]:
                    node[2][d] = new
                    break
                node = node[2][d]
    write_csv(work / 'near_duplicates.csv', ['left', 'right', 'distance', 'decision'], candidates)
    (work / 'classes.txt').write_text('\n'.join(NAMES) + '\n', encoding='utf-8')
    write_json(work / 'inventory.json', {
        'raw_images_by_source': dict(inventory), 'unique_images': len(records),
        'exact_duplicates_removed': sum(len(r['origins'])-1 for r in records),
        'unreadable': errors, 'near_duplicate_candidates': len(candidates),
        'phash_distance': distance,
        'annotation_files': [p.relative_to(raw).as_posix() for p in sorted(raw.rglob('*'))
                             if p.suffix.lower() in {'.txt', '.xml', '.json', '.yaml', '.csv'}],
        'warning': 'Original annotations are not person labels. Video groups require manual review.'})
    print(f'Prepared {len(records)} images in {work}. Complete review.csv and labels before export.')


def read_labels(path):
    if not path.is_file():
        raise ValueError(f'Missing label: {path.name}')
    labels = []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] not in {'0', '1'}:
            raise ValueError(f'{path.name}:{line_number}: expected class 0/1 and four coordinates')
        x, y, w, h = map(float, parts[1:])
        if (not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0
                or x-w/2 < -1e-6 or y-h/2 < -1e-6
                or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6):
            raise ValueError(f'{path.name}:{line_number}: invalid normalized box')
        labels.append((int(parts[0]), x, y, w, h))
    return labels


def collect(work):
    records = json.loads((work / 'manifest.json').read_text(encoding='utf-8'))
    ids = {r['id'] for r in records}
    reviews = read_csv(work / 'review.csv')
    if len(reviews) != len(ids) or {r['id'] for r in reviews} != ids:
        raise ValueError('review.csv must contain exactly one row for every manifest image')
    reviews = {r['id']: r for r in reviews}
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    groups, errors, accepted, excluded = {}, [], [], []
    for record in records:
        key = record['id']
        review = reviews[key]
        if review['status'] not in {'pending', 'approved', 'excluded'}:
            errors.append(f'{key}: invalid review status')
        group = review['group'].strip()
        if group:
            if group in groups:
                union(key, groups[group])
            groups[group] = key
        if review['status'] != 'approved':
            excluded.append({'id': key, 'status': review['status'], 'notes': review['notes']})
            continue
        try:
            with Image.open(work / 'images' / f'{key}.png') as im:
                if pixel_hash(im.convert('RGB')) != record['pixel_hash']:
                    raise ValueError('Image changed after preparation')
            labels = read_labels(work / 'labels' / f'{key}.txt')
            accepted.append({**record, 'labels': labels, 'review': review})
        except (OSError, ValueError) as error:
            errors.append(f'{key}: {error}')
    for pair in read_csv(work / 'near_duplicates.csv'):
        if pair['left'] not in ids or pair['right'] not in ids:
            errors.append('Unknown image in near_duplicates.csv')
        elif pair['decision'] in {'same', 'pending'}:
            # Conservative grouping protects splits even when pair review is unfinished.
            union(pair['left'], pair['right'])
        elif pair['decision'] != 'different':
            errors.append('Near-duplicate decision must be pending, same, or different')
    for record in accepted:
        record['group'] = find(record['id'])
    report = {'accepted': len(accepted), 'excluded': excluded, 'errors': errors,
              'images_without_known_video_group': sum(not r['review']['group'].strip() for r in accepted),
              'warning': 'Unknown video provenance and undetected near duplicates may cause leakage.'}
    write_json(work / 'validation.json', report)
    if errors:
        raise ValueError(f'{len(errors)} validation errors; see {work / "validation.json"}')
    if not accepted:
        raise ValueError('No approved images. Complete manual annotation and review first.')
    return accepted, report


def assign_splits(records, seed):
    grouped = defaultdict(list)
    for r in records:
        grouped[r['group']].append(r)
    if len(grouped) < 3:
        raise ValueError('At least three independent groups are required for three nonempty splits')

    def features(rows):
        result = Counter(images=len(rows))
        for r in rows:
            result.update(f'source:{s}' for s in {o['source'] for o in r['origins']})
            result.update(f'class:{label[0]}' for label in r['labels'])
            if not r['labels']:
                result['empty'] += 1
        return result

    totals = features(records)
    items = sorted(grouped.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: -len(item[1]))
    ratios = {'train': .8, 'valid': .1, 'test': .1}
    counts = {s: Counter() for s in ratios}
    allocation, used = {}, set()
    for index, (group, rows) in enumerate(items):
        vector = features(rows)
        empty = set(ratios) - used
        choices = [s for s in ratios if s in empty] if len(items)-index == len(empty) else list(ratios)

        def cost(split):
            return sum(((counts[split][k]+v-totals[k]*ratios[split])**2
                        - (counts[split][k]-totals[k]*ratios[split])**2)
                       / max(totals[k]*ratios[split], 1) for k, v in vector.items())

        chosen = min(choices, key=cost)
        counts[chosen].update(vector)
        allocation[group] = chosen
        used.add(chosen)
    return allocation, counts


def export(work, output, yaml_path, seed):
    if output.exists() or yaml_path.exists():
        raise ValueError('Output or YAML already exists. Choose new --output and --yaml paths.')
    records, report = collect(work)
    allocation, counts = assign_splits(records, seed)
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'valid', 'test'):
        for folder in ('images', 'labels', 'previews'):
            (output / split / folder).mkdir(parents=True)
    preview_counts = Counter()
    for r in records:
        split, key = allocation[r['group']], r['id']
        r['split'] = split
        image_path = work / 'images' / f'{key}.png'
        shutil.copy2(image_path, output / split / 'images' / image_path.name)
        label_path = work / 'labels' / f'{key}.txt'
        shutil.copy2(label_path, output / split / 'labels' / label_path.name)
        r['label_sha256'] = hashlib.sha256(label_path.read_bytes()).hexdigest()
        buckets = [(split, o['source']) for o in r['origins']]
        if any(preview_counts[b] < 3 for b in buckets):
            with Image.open(image_path) as im:
                im = im.convert('RGB')
                draw = ImageDraw.Draw(im)
                for cls, x, y, w, h in r['labels']:
                    box = ((x-w/2)*im.width, (y-h/2)*im.height,
                           (x+w/2)*im.width, (y+h/2)*im.height)
                    draw.rectangle(box, outline='red' if cls else 'lime', width=3)
                    draw.text((box[0], box[1]), NAMES[cls], fill='red' if cls else 'lime')
                im.save(output / split / 'previews' / image_path.name)
            preview_counts.update(buckets)
    write_json(output / 'manifest.json', records)
    report.update(seed=seed, requested_ratios={'train': .8, 'valid': .1, 'test': .1},
                  actual_counts={s: dict(c) for s, c in counts.items()},
                  actual_ratios={s: c['images']/len(records) for s, c in counts.items()})
    write_json(output / 'report.json', report)
    # Keep the full review evidence with the frozen export.
    for name in ('review.csv', 'near_duplicates.csv', 'inventory.json'):
        shutil.copy2(work / name, output / name)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f'path: {json.dumps(output.resolve().as_posix())}\n'
                         'train: train/images\nval: valid/images\ntest: test/images\n'
                         'names:\n  0: normal_person\n  1: potentially_dangerous_person\n', encoding='utf-8')
    print(f'Exported {len(records)} reviewed images to {output}; config: {yaml_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'export'])
    parser.add_argument('--raw', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--work', type=Path, default=ROOT / 'data/annotation')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/dataset')
    parser.add_argument('--yaml', type=Path, default=ROOT / 'config/data.yaml')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--phash-distance', type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.phash_distance <= 64:
        parser.error('--phash-distance must be between 0 and 64')
    try:
        if args.command == 'prepare':
            prepare(args.raw, args.work, args.phash_distance)
        elif args.command == 'validate':
            _, report = collect(args.work)
            print(json.dumps(report, indent=2))
        else:
            export(args.work, args.output, args.yaml, args.seed)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
