import os
import glob
import collections
import shutil
import numpy as np
from collections import defaultdict

# Important: relations_to_run.tsv determines what relations will be tested.


class Split:
    """
    This class manages data split (train/test/dev).

    :ivar relation_to_splits_to_instances: an ordered dict mapping from a relation to data in each split.
        Data in each split is stored as a list of tuples, where each tuple is (subj, obj, label)
    """

    def __init__(self):
        # a map from a relation to maps named "train"/"test"/"dev" to instances in corresponding split
        self.relation_to_splits_to_instances = collections.OrderedDict()

    def read_splits(self, split_directory, vocabs, entity_name_is_typed,
                    create_development_set_if_not_exist=False):
        """
        This function initializes data split from the files in the split directory. The split directory follows the
        format of PRA split.

        :param split_directory: where split files are stored
        :param vocabs: vocabs of entities and relations
        :param entity_name_is_typed: whether entities has type prefixes
        :param create_development_set_if_not_exist: if set to True, a development set will be created by randomly
            sampling 20% of training instances.
        :type vocabs: :meth:`main.data.Vocabs`

        .. note::

            The split directory contains ``relation_to_run.tsv``, ``params.json``, and a folder for each relation.

            - relation_to_run.tsv: determines what relations will be tested.
            - params.json: used by PRA code to store split meta-information, such as positive to negative example ratio
            - each relation's folder: contains ``training.tsv``, ``testing.tsv``, and ``development.tsv``

        """
        # Improvement: handle entity name without types using flag entity_name_is_typed. possibly using relation domain
        #              and range to augment entity name.

        # 1. Read
        relations_filename = os.path.join(split_directory, "relations_to_run.tsv")
        with open(relations_filename, "r") as fh:
            for line in fh:
                rel = line.strip()
                if rel not in vocabs.relation_to_idx:
                    raise Exception(rel, "is not in relation vocabulary")
                self.relation_to_splits_to_instances[rel] = {}

        train_entities = defaultdict(set)
        test_entities = defaultdict(set)
        development_set_exists = False
        for rel in self.relation_to_splits_to_instances:
            rel_dir = os.path.join(split_directory, rel)
            split_filenames = [f for f in glob.glob(rel_dir+"/*.tsv")]
            for split_filename in split_filenames:
                instances = []
                with open(split_filename) as fh:
                    for line in fh:
                        subj, obj, label = line.strip().split("\t")
                        label = int(label)
                        if subj not in vocabs.node_to_idx:
                            raise Exception(subj, "is not in entity vocabulary")
                        if obj not in vocabs.node_to_idx:
                            raise Exception(obj, "is not in entity vocabulary")
                        assert(label == 1 or label == -1)
                        instances.append(tuple([subj, obj, label]))
                        # used to calculate how many new entities are in the testing set
                        if "training" in split_filename:
                            train_entities[rel].add(subj)
                            train_entities[rel].add(obj)
                        if "testing" in split_filename:
                            test_entities[rel].add(subj)
                            test_entities[rel].add(obj)

                split = split_filename.split("/")[-1].split(".")[0]
                assert(split == "training" or split == "testing" or split == "development")
                if split == "development":
                    development_set_exists = True
                self.relation_to_splits_to_instances[rel][split] = instances

        for rel in train_entities:
            new_entities_in_testing = test_entities[rel] - train_entities[rel]
            print("Relation {}: {}% new entities in testing".format(rel, len(new_entities_in_testing) * 100.0 / len(test_entities[rel])))

        # 2. Create development set
        if create_development_set_if_not_exist and not development_set_exists:
            print("Creating development set by splitting training set.")
            print("Current train/test split in folder will be replaced with new train/dev/test split")
            if split_directory[-1] == "/":
                split_directory = split_directory[:-1]
            tmp_split_directory = split_directory + "_TMP"
            assert not os.path.exists(tmp_split_directory)
            os.mkdir(tmp_split_directory)

            shutil.copy(os.path.join(split_directory, "params.json"), tmp_split_directory)
            shutil.copy(os.path.join(split_directory, "relations_to_run.tsv"), tmp_split_directory)

            # split development set from training set
            for rel in self.relation_to_splits_to_instances:
                training_instances = self.relation_to_splits_to_instances[rel]["training"]
                np.random.shuffle(training_instances)
                pos = int(len(training_instances) * 0.8)
                new_training_instances = training_instances[:pos]
                development_instances = training_instances[pos:]
                self.relation_to_splits_to_instances[rel]["training"] = new_training_instances
                self.relation_to_splits_to_instances[rel]["development"] = development_instances

            # write new split
            for rel in self.relation_to_splits_to_instances:
                rel_dir = os.path.join(tmp_split_directory, rel)
                if not os.path.exists(rel_dir):
                    os.mkdir(rel_dir)
                for spt in self.relation_to_splits_to_instances[rel]:
                    split_filename = os.path.join(rel_dir, spt + ".tsv")
                    with open(split_filename, "w+") as fh:
                        for subj, obj, label in self.relation_to_splits_to_instances[rel][spt]:
                            fh.write(subj + "\t" + obj + "\t" + str(label) + "\n")

            # replace split_dir with new contents
            shutil.rmtree(split_directory)
            shutil.copytree(tmp_split_directory, split_directory)
            shutil.rmtree(tmp_split_directory)

        train_numbers = []
        test_numbers = []
        for rel in self.relation_to_splits_to_instances:
            train_numbers.append(len(self.relation_to_splits_to_instances[rel]["training"]))
            test_numbers.append(len(self.relation_to_splits_to_instances[rel]["testing"]))
        print("Avg. # training instances:", sum(train_numbers) / len(train_numbers))
        print("Avg. # testing instances:", sum(test_numbers) / len(test_numbers))








