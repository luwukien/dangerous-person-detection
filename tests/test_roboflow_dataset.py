import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from scripts.prepare_dataset import collect, export, read_csv, write_csv
from scripts.roboflow_dataset import import_coco, prepare_upload, resolve_image


def png(color):
    stream = io.BytesIO()
    Image.new('RGB', (40, 60), color).save(stream, format='PNG')
    return stream.getvalue()


class RoboflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / 'raw'
        source = self.raw / 'owner/source'
        source.mkdir(parents=True)
        (source / 'same.png').write_bytes(png('red'))
        (source / 'broken.jpg').write_bytes(b'broken')
        (source / 'video.mp4').write_bytes(b'video')
        (source / 'partial.zip.crdownload').write_bytes(b'partial')
        (source / 'bad.zip').write_bytes(b'broken zip')
        (source / 'download-manifest.json').write_text('{"revision":"test"}')
        with zipfile.ZipFile(source / 'images.zip', 'w') as z:
            z.writestr('copy.png', png('red'))
            z.writestr('same.png', png('green'))
            z.writestr('camera_Segment_1_x264_frame_1.png', png('blue'))
            z.writestr('../unsafe.png', png('yellow'))
            z.writestr('labels.xml', '<weapon/>')
        self.work = self.root / 'work'
        prepare_upload(self.raw, self.work, batch_size=2)
        self.records = json.loads((self.work / 'manifest.json').read_text())
        self.export_dir = self.root / 'coco'
        self.export_dir.mkdir()
        self.coco = {'categories': [{'id': 8, 'name': 'potentially_dangerous_person'},
                                    {'id': 3, 'name': 'normal_person'},
                                    {'id': 0, 'name': 'unused-parent'}],
                     'images': [{'id': i, 'file_name': r['id'] + '_png.rf.abc123.jpg',
                                 'width': 40, 'height': 60} for i, r in enumerate(self.records)],
                     'annotations': [{'id': 1, 'image_id': 0, 'category_id': 8,
                                      'bbox': [4, 6, 20, 30]},
                                     {'id': 2, 'image_id': 1, 'category_id': 3,
                                      'bbox': [0, 0, 40, 60]}]}

    def save_coco(self, data=None, folder=None):
        folder = folder or self.export_dir
        folder.mkdir(parents=True, exist_ok=True)
        (folder / '_annotations.coco.json').write_text(json.dumps(data or self.coco))

    def test_archives_provenance_and_batches(self):
        self.assertEqual(len(self.records), 3)
        self.assertEqual(sum(len(r['origins']) for r in self.records), 4)
        report = json.loads((self.work / 'upload-manifest.json').read_text())
        packed = []
        for batch in report['batches']:
            with zipfile.ZipFile(self.work / batch['path']) as z:
                self.assertLessEqual(len(z.namelist()), 2)
                self.assertTrue(all(n.endswith('.png') for n in z.namelist()))
                self.assertIsNone(z.testzip())
                packed.extend(z.namelist())
        self.assertEqual(len(packed), len(set(packed)))
        self.assertEqual(len(packed), 3)
        inventory = json.loads((self.work / 'inventory.json').read_text())
        self.assertEqual(len(inventory['unreadable']), 3)
        self.assertTrue(any('partial' in r['path'] for r in inventory['skipped']))
        self.assertTrue(any('::labels.xml' in p for p in inventory['annotation_files']))
        self.assertFalse((self.root / 'unsafe.png').exists())
        hints = read_csv(self.work / 'group_suggestions.csv')
        self.assertTrue(any(r['suggested_group'].endswith('/video/camera') for r in hints))
        self.assertTrue(json.loads((self.work / 'source_provenance.json').read_text())['notices'])
        with self.assertRaises(FileExistsError):
            prepare_upload(self.raw, self.work)

    def test_import_classes_empty_labels_and_review(self):
        self.save_coco()
        report = import_coco(self.work, self.export_dir)
        self.assertEqual(report['imported_images'], 3)
        self.assertEqual(len(report['empty_label_candidates']), 1)
        self.assertEqual((self.work / 'labels' / (self.records[0]['id'] + '.txt')).read_text(),
                         '1 0.35 0.35 0.5 0.5\n')
        self.assertTrue(all(r['status'] == 'pending' for r in read_csv(self.work / 'review.csv')))
        self.assertEqual(len(list((self.work / report['previews']).glob('*.png'))), 3)
        import_coco(self.work, self.export_dir)  # Idempotent import.
        with self.assertRaisesRegex(ValueError, 'No approved'):
            collect(self.work)
        reviews = read_csv(self.work / 'review.csv')
        for row in reviews:
            row['status'] = 'approved'
        write_csv(self.work / 'review.csv', ['id', 'status', 'group', 'notes'], reviews)
        with self.assertRaisesRegex(ValueError, 'validation errors'):
            collect(self.work)
        for row in reviews:
            row['group'] = 'confirmed-independent-' + row['id']
        write_csv(self.work / 'review.csv', ['id', 'status', 'group', 'notes'], reviews)
        pairs = read_csv(self.work / 'near_duplicates.csv')
        for pair in pairs:
            pair['decision'] = 'different'
        write_csv(self.work / 'near_duplicates.csv', ['left', 'right', 'distance', 'decision'], pairs)
        output = self.root / 'dataset'
        export(self.work, output, self.root / 'data.yaml', 42)
        self.assertEqual({r['split'] for r in json.loads((output / 'manifest.json').read_text())},
                         {'train', 'valid', 'test'})
        self.assertTrue((output / 'source_provenance.json').exists())

    def test_invalid_import_does_not_write_labels(self):
        bad = []
        for field, value in [('bbox', [-1, 0, 10, 10]), ('bbox', [0, 0, float('nan'), 10]),
                             ('bbox', [0, 0, 41, 10]), ('category_id', 99),
                             ('image_id', 100), ('iscrowd', 1)]:
            data = copy.deepcopy(self.coco)
            data['annotations'][0][field] = value
            bad.append(data)
        for field, value in [('width', 20), ('file_name', 'unknown.jpg')]:
            data = copy.deepcopy(self.coco)
            data['images'][0][field] = value
            bad.append(data)
        data = copy.deepcopy(self.coco)
        data['images'].append(data['images'][0])
        bad.append(data)
        for data in bad:
            with self.subTest(data=data):
                self.save_coco(data)
                with self.assertRaises(ValueError):
                    import_coco(self.work, self.export_dir)
                self.assertFalse(list((self.work / 'labels').iterdir()))

    def test_partial_import_missing_is_not_negative_and_conflicts_block(self):
        data = copy.deepcopy(self.coco)
        data['images'] = data['images'][:1]
        data['annotations'] = data['annotations'][:1]
        self.save_coco(data, self.export_dir / 'train')
        report = import_coco(self.work, self.export_dir)
        self.assertEqual(len(report['missing_from_this_export']), 2)
        self.assertEqual(len(list((self.work / 'labels').glob('*.txt'))), 1)
        different = copy.deepcopy(data)
        different['annotations'][0]['category_id'] = 3
        self.save_coco(different, self.export_dir / 'valid')
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            import_coco(self.work, self.export_dir)
        (self.export_dir / 'valid' / '_annotations.coco.json').unlink()
        self.save_coco(different, self.export_dir / 'train')
        with self.assertRaisesRegex(ValueError, 'existing labels differ'):
            import_coco(self.work, self.export_dir)

    def test_name_mapping_and_modified_local_image(self):
        records = {r['id']: r for r in self.records}
        key = self.records[0]['id']
        for name in [f'{key}.png', f'{key}.rf.123.jpg', f'{key}_png.rf.123.jpg']:
            self.assertEqual(resolve_image(name, records), key)
        with self.assertRaises(ValueError):
            resolve_image(key + '_other_image.png', records)
        self.save_coco()
        (self.work / 'images' / f'{key}.png').write_bytes(png('yellow'))
        with self.assertRaisesRegex(ValueError, 'local image changed'):
            import_coco(self.work, self.export_dir)
        self.assertFalse(list((self.work / 'labels').iterdir()))

    def test_exif_orientation_and_oversize_are_accounted_for(self):
        raw = self.root / 'other-raw'
        raw.mkdir()
        im = Image.new('RGB', (20, 30), 'purple')
        exif = Image.Exif()
        exif[274] = 6
        im.save(raw / 'rotated.jpg', exif=exif)
        Image.new('RGB', (40, 11000), 'green').save(raw / 'too-tall.png')
        work = self.root / 'other-work'
        report = prepare_upload(raw, work)
        self.assertEqual(report['prepared_unique'], 2)
        self.assertEqual(report['upload_ready'], 1)
        self.assertEqual(len(report['blocked']), 1)
        manifest = json.loads((work / 'manifest.json').read_text())
        rotated = next(r for r in manifest if r['origins'][0]['path'] == 'rotated.jpg')
        self.assertEqual((rotated['width'], rotated['height']), (30, 20))
        self.assertEqual(len(list((work / 'images').glob('*.png'))), 2)
        blocked = report['blocked'][0]['id']
        self.assertFalse(any(blocked in b['ids'] for b in report['batches']))


if __name__ == '__main__':
    unittest.main()
