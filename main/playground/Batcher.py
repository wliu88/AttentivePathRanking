import torch


class Batcher:
    def __init__(self, filename, batch_size, shuffle):
        self.labels = None
        self.inputs = None
        self.read_data(filename)
        self.number_entity_pairs, self.number_of_paths, self.path_length, self.feature_size = self.inputs.shape

        self.shuffle = shuffle
        if shuffle:
            self.shuffle_data()

        # how many entity pairs will be bundled together
        self.batch_size = batch_size

        # used to point to the current entity pair
        self.current_index = 0

    def read_data(self, filename):
        with open(filename, "r") as fh:
            inputs = []
            labels = []
            for line in fh:
                line = line.strip()
                if len(line) != 0:
                    paths_for_pair = []
                    label, paths = line.split("\t")
                    label = int(label)
                    labels.append(label)
                    paths = paths.split(";")
                    for path in paths:
                        whole_path_features = []
                        # a token can be a index or a list of indices representing a relation, entity, or entity types
                        steps = path.split(" ")
                        for step in steps:
                            features = step.split(",")
                            features = [int(f) for f in features]
                            whole_path_features.append(features)
                        paths_for_pair.append(whole_path_features)
                    inputs.append(paths_for_pair)
        self.inputs = torch.LongTensor(inputs)
        self.labels = torch.FloatTensor(labels)
        # print(self.inputs.shape)
        # print(self.labels.shape)

    def shuffle_data(self):
        # only long type or byte type tensor can be used for index
        indices = torch.randperm(self.number_entity_pairs).long()
        self.inputs = self.inputs[indices]
        self.labels = self.labels[indices]

    def get_batch(self):
        start_index = self.current_index
        if start_index >= self.number_entity_pairs:
            return None
        end_index = min(start_index+self.batch_size-1, self.number_entity_pairs-1)
        batch_inputs = self.inputs[start_index:end_index+1]
        batch_labels = self.labels[start_index:end_index+1]
        self.current_index = end_index + 1
        return batch_inputs, batch_labels

    def reset(self):
        self.current_index = 0
        if self.shuffle:
            self.shuffle_data()

    def get_size(self):
        return self.number_entity_pairs, self.number_of_paths, self.path_length, self.feature_size


if __name__ == "__main__":
    batcher = Batcher("/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/_architecture_structure_address/train/train.txt.2.int", 3, False)
    finished = False
    count = 0
    while not finished:
        data = batcher.get_batch()
        if data is None:
            break
        inputs, labels = data
        print(labels.shape)
        print(inputs.shape)
        count += 1
    print(count)

