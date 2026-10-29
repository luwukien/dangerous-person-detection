import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from PIL import Image

from scripts.prepare_dataset import pixel_hash
from scripts.select_roboflow_upload import choose, select_upload


class SelectRoboflowUploadTests(unittest.TestCase):
    def test_repeatable_source_balance_and_whole_groups(self):
        records = [{'id': str(i), 'source': 'a' if i < 6 else 'b',
                    'group': 'video' if i in (0, 1, 2) else ''} for i in range(10)]
        first = choose(records, 5, 42)
        self.assertEqual(first, choose(records, 5, 42))
        self.assertLessEqual(len(first), 5)
        self.assertIn(len([r for r in first if r['id'] in {'0', '1', '2'}]), {0, 3})

    def test_images_only_zip_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            images = work / 'images'
            images.mkdir(parents=True)
            origins = []
            for i in range(6):
                image = Image.new('RGB', (12, 12), (i * 30, 10, 20))
                key = pixel_hash(image)
                image.save(images / f'{key}.png')
                origins.append({'id': key, 'origins': [{'source': 'test/source',
                                                       'path': f'frame-{i}.png'}]})
            (work / 'manifest.json').write_text(json.dumps(origins), encoding='utf-8')
            output = root / 'upload-selected'
            preview = select_upload(work, root / 'raw', output, max_images=5,
                                    fraction=0.5, batch_size=2, preview=True)
            self.assertEqual(preview['selected_images'], 3)
            self.assertFalse(output.exists())
            report = select_upload(work, root / 'raw', output, max_images=5,
                                   fraction=0.5, batch_size=2)
            self.assertEqual(report['selected_images'], 3)
            self.assertEqual(report['unmatched_provenance'], 0)
            self.assertEqual(len(report['batches']), 2)
            packed = []
            for batch in report['batches']:
                with zipfile.ZipFile(output / batch['path']) as archive:
                    self.assertIsNone(archive.testzip())
                    packed.extend(archive.namelist())
            self.assertEqual(len(packed), 3)
            self.assertTrue(all(name.endswith('.png') for name in packed))
            with self.assertRaises(FileExistsError):
                select_upload(work, root / 'raw', output)


if __name__ == '__main__':
    unittest.main()
