import unittest
from main.playground.BatcherFileList import BatcherFileList
from tqdm import tqdm


class TestBatcherFileList(unittest.TestCase):
    def setUp(self):
        # need to specify correct absolute path to data
        self.files_dir = "data/wordnet18rr/cvsm_entity/data/auto_generated_data_output/also_see/dev"

    def test_shuffled_iterations(self):
        batcher = BatcherFileList(self.files_dir, batch_size=32, shuffle=True, max_number_batchers_on_gpu=100)
        count = 0
        while True:
            data = batcher.get_batch()
            if data is None:
                break
            count += 1

        count1 = 0
        for i in tqdm(range(0, count)):
            data = batcher.get_batch()
            count1 += 1

        assert count == count1
        assert batcher.get_batch() is None
        assert batcher.get_batch() is not None

    def test_deterministic_iterations(self):
        batcher = BatcherFileList(self.files_dir, batch_size=100, shuffle=False, max_number_batchers_on_gpu=100)
        list_path_numbers = []
        while True:
            data = batcher.get_batch()
            if data is None:
                break
            list_path_numbers.append(data[0].shape[1])

        list_path_numbers1 = []
        while True:
            data = batcher.get_batch()
            if data is None:
                break
            list_path_numbers1.append(data[0].shape[1])
        assert list_path_numbers == list_path_numbers1







