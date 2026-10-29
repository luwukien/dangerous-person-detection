import unittest

from download_datasets import dataset_handle


class DatasetURLTests(unittest.TestCase):
    def test_dataset_page_and_data_tab_have_same_handle(self):
        handle = 'trainingdatapro/people-with-guns-segmentation-and-detection'
        base = f'https://www.kaggle.com/datasets/{handle}'
        for suffix in ('', '/', '/data', '/data/', '/data?select=images.zip#files'):
            with self.subTest(suffix=suffix):
                self.assertEqual(dataset_handle(base + suffix), handle)

    def test_whitespace_and_dataset_named_data(self):
        self.assertEqual(dataset_handle('  https://kaggle.com/datasets/owner/data/data  '),
                         'owner/data')

    def test_rejects_invalid_hosts_paths_and_extra_segments(self):
        for url in (
            'https://evil.com/datasets/owner/name/data',
            'https://kaggle.com.evil.com/datasets/owner/name/data',
            'https://kaggle.com/datasets/owner',
            'https://kaggle.com/datasets/owner/../data',
            'https://kaggle.com/datasets/owner/name/data/extra',
            'https://kaggle.com/datasets/owner/name/versions/2',
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                dataset_handle(url)


if __name__ == '__main__':
    unittest.main()
