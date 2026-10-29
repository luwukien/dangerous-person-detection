import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from download_datasets import dataset_handle
from scripts.prepare_dataset import (assign_splits, collect, export, prepare,
                                     read_csv, read_labels, write_csv)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        self.work = self.root / 'work'
        for index, color in enumerate(['red', 'green', 'blue', 'yellow', 'purple', 'orange']):
            folder = self.raw / 'owner' / f'source{index % 2}'
            folder.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (40, 60), color).save(folder / f'{index}.png')
        # Identical content from another source must retain both origins.
        Image.new('RGB', (40, 60), 'red').save(self.raw / 'owner/source1/copy.png')
        prepare(self.raw, self.work, 4)
        self.records = json.loads((self.work / 'manifest.json').read_text())

    def approve(self):
        write_csv(self.work / 'review.csv', ['id', 'status', 'group', 'notes'],
                  [{'id': r['id'], 'status': 'approved', 'group': '', 'notes': 'checked'}
                   for r in self.records])
        for r in self.records:
            (self.work / 'labels' / f'{r["id"]}.txt').write_text('0 0.5 0.5 0.5 0.8\n1 0.2 0.2 0.1 0.1\n')
        pairs = read_csv(self.work / 'near_duplicates.csv')
        for pair in pairs:
            pair['decision'] = 'different'
        write_csv(self.work / 'near_duplicates.csv', ['left', 'right', 'distance', 'decision'], pairs)

    def test_dedup_and_pending_gate(self):
        self.assertEqual(len(self.records), 6)
        self.assertEqual(sum(len(r['origins']) for r in self.records), 7)
        with self.assertRaisesRegex(ValueError, 'No approved'):
            collect(self.work)

    def test_missing_label_is_not_negative(self):
        self.approve()
        (self.work / 'labels' / f'{self.records[0]["id"]}.txt').unlink()
        with self.assertRaisesRegex(ValueError, 'validation errors'):
            collect(self.work)

    def test_empty_label_and_invalid_coordinates(self):
        path = self.root / 'label.txt'
        path.write_text('')
        self.assertEqual(read_labels(path), [])
        for text in ['2 .5 .5 .1 .1', '0 nan .5 .1 .1', '0 .9 .5 .5 .5', '1 .5 .5 0 .2']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                read_labels(path)

    def test_image_mutation_detected(self):
        self.approve()
        Image.new('RGB', (40, 60), 'black').save(self.work / 'images' / f'{self.records[0]["id"]}.png')
        with self.assertRaises(ValueError):
            collect(self.work)

    def test_group_and_split_reproducibility(self):
        self.approve()
        rows = read_csv(self.work / 'review.csv')
        rows[0]['group'] = rows[1]['group'] = 'video-1'
        write_csv(self.work / 'review.csv', ['id', 'status', 'group', 'notes'], rows)
        records, _ = collect(self.work)
        self.assertEqual(records[0]['group'], records[1]['group'])
        allocation, _ = assign_splits(records, 42)
        self.assertEqual(allocation, assign_splits(records, 42)[0])
        self.assertEqual(set(allocation.values()), {'train', 'valid', 'test'})

    def test_pending_similar_pairs_cannot_leak(self):
        self.approve()
        write_csv(self.work / 'near_duplicates.csv', ['left', 'right', 'distance', 'decision'],
                  [{'left': self.records[0]['id'], 'right': self.records[1]['id'],
                    'distance': 0, 'decision': 'pending'}])
        records, _ = collect(self.work)
        self.assertEqual(records[0]['group'], records[1]['group'])

    def test_export_and_no_overwrite(self):
        self.approve()
        output, yaml_path = self.root / 'dataset', self.root / 'data.yaml'
        export(self.work, output, yaml_path, 42)
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(len(manifest), 6)
        self.assertIn('val: valid/images', yaml_path.read_text())
        for row in manifest:
            self.assertTrue((output / row['split'] / 'labels' / f'{row["id"]}.txt').is_file())
        with self.assertRaises(ValueError):
            export(self.work, output, yaml_path, 42)

    def test_insufficient_groups_block_export(self):
        self.approve()
        rows = read_csv(self.work / 'review.csv')
        for row in rows:
            row['group'] = 'same-video'
        write_csv(self.work / 'review.csv', ['id', 'status', 'group', 'notes'], rows)
        records, _ = collect(self.work)
        with self.assertRaisesRegex(ValueError, 'three independent'):
            assign_splits(records, 42)

    def test_known_box_and_verified_negative(self):
        self.approve()
        path = self.work / 'labels' / f'{self.records[0]["id"]}.txt'
        path.write_text('')
        records, _ = collect(self.work)
        self.assertEqual(records[0]['labels'], [])
        self.assertEqual(records[1]['labels'][0], (0, .5, .5, .5, .8))

    def test_handle_validation(self):
        self.assertEqual(dataset_handle('https://www.kaggle.com/datasets/owner/data?x=1'), 'owner/data')
        for url in ['https://evil.com/datasets/a/b', 'https://kaggle.com/datasets/a/..']:
            with self.assertRaises(ValueError):
                dataset_handle(url)


if __name__ == '__main__':
    unittest.main()
