from main.playground.Batcher import Batcher
import torch
import os

# Debug: Not finished

class BatcherFileList:
    def __init__(self, data_dir, batch_size, shuffle, max_number_batchers_on_gpu):
        self.do_shuffle = shuffle
        self.batch_size = batch_size

        # batchers store all batchers
        self.batchers = []
        self.initialize_batchers(data_dir)
        self.number_batchers_on_gpu = min(max_number_batchers_on_gpu, len(self.batchers))
        if self.do_shuffle:
            self.shuffle_batchers()

        self.current_index = 0
        self.current_gpu_index = 0
        self.empty_batcher_indices = set()

        self.gpu_labels = []
        self.gpu_inputs = []
        self.preallocate_gpu()

    def initialize_batchers(self, data_dir):
        print("Reading files from", data_dir)
        for file in os.listdir(data_dir):
            if file[-3:] == "int":
                self.batchers.append(Batcher(os.path.join(data_dir, file), self.batch_size, self.do_shuffle))

    def preallocate_gpu(self):
        """
        Preallocate gpu space for data from current indexed batcher to the batcher that makes the total number of
        batchers on gpu equal to number_batchers_on_gpu
        :return:
        """
        self.gpu_labels = []
        self.gpu_inputs = []
        # Important: min(self.current_index + self.number_batchers_on_gpu, len(self.batchers)) is used to deal with
        #            the last group of batchers that may be less than number_batchers_on_gpu.
        #            e.g., for example, when we have 100 batchers, the number_batchers_on_gpu is 30, we need to deal
        #            the last 10 batchers.
        for i in range(self.current_index, min(self.current_index + self.number_batchers_on_gpu, len(self.batchers))):
            batcher = self.batchers[i]
            number_entity_pairs, number_of_paths, path_length, feature_size = batcher.get_size()
            # here we create gpu tensors of specified dimensions
            self.gpu_inputs.append(torch.cuda.LongTensor(self.batch_size, number_of_paths, path_length, feature_size))
            self.gpu_labels.append(torch.cuda.FloatTensor(self.batch_size, 1))
        self.populate_gpu()

    def populate_gpu(self):
        for i in range(self.current_index, min(self.current_index + self.number_batchers_on_gpu, len(self.batchers))):
            # current batch was alreday finished
            if i in self.empty_batcher_indices:
                continue

            batcher = self.batchers[i]
            data = batcher.get_batch()
            # current batch is finished
            if data is None:
                self.empty_batcher_indices.add(i)
                continue

            # copy data from cpu to gpu
            inputs, labels = data
            self.gpu_inputs[i % self.number_batchers_on_gpu].resize_(inputs.shape).copy_(inputs)
            self.gpu_labels[i % self.number_batchers_on_gpu].resize_(labels.shape).copy_(labels)

    def shuffle_batchers(self):
        shuffled_batchers = []
        for i in torch.randperm(len(self.batchers)):
            shuffled_batchers.append(self.batchers[i])
        self.batchers = shuffled_batchers

    def get_batch(self):
        # Important: the outer loop is to iterate through all data.
        #            the inner loop is to iterate through current group of batchers we preallocate gpu space for.
        while len(self.empty_batcher_indices) < len(self.batchers):
            # empty_batcher_indices is for all batchers
            # print(len(self.empty_batcher_indices), self.number_batchers_on_gpu + self.current_index)
            while len(self.empty_batcher_indices) < min(self.current_index + self.number_batchers_on_gpu, len(self.batchers)):
                # one loop through batchers on gpu has finished. This does not mean these batchers are used up.
                # It just means we need to get new data from these batchers.
                if self.current_gpu_index >= self.number_batchers_on_gpu or self.current_gpu_index + self.current_index >= len(self.batchers):
                    self.populate_gpu()
                    self.current_gpu_index = 0

                # current batcher was already finished
                if self.current_index + self.current_gpu_index in self.empty_batcher_indices:
                    self.current_gpu_index += 1
                    continue

                # return the content from the current batcher
                inputs, labels = self.gpu_inputs[self.current_gpu_index], self.gpu_labels[self.current_gpu_index]
                self.current_gpu_index += 1
                return inputs, labels
            # batchers on gpu has all been used up
            if len(self.empty_batcher_indices) < len(self.batchers):
                self.current_index = self.current_index + self.number_batchers_on_gpu
                self.preallocate_gpu()
                self.current_gpu_index = 0
        # end of an epoch
        self.reset()
        return None

    def reset(self):
        self.current_index = 0
        self.current_gpu_index = 0
        self.empty_batcher_indices = set()
        if self.do_shuffle:
            self.shuffle_batchers()
        for batcher in self.batchers:
            batcher.reset()
        self.preallocate_gpu()