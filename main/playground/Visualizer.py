import os
import numpy as np
import shutil
import pickle


class Visualizer:

    def __init__(self, idx2entity, idx2entity_type, idx2relation, save_dir, mid2name_filename=None):
        self.idx2entity = idx2entity
        self.idx2entity_type = idx2entity_type
        self.idx2relation = idx2relation

        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.mid2name = None
        if mid2name_filename is not None:
            self.mid2name = pickle.load(open(mid2name_filename, "rb"))

        # this is a dictionary from query relation to another dictionary mapping from relation paths to contradictions
        self.rel_path2contradictions = {}

    def visualize_paths(self, inputs, labels, type_weights, path_weights, rel, split, epoch,
                        filter_negative_example=False, filter_false_prediction=False, probs=None,
                        top_k_path=None, minimal_path_weight=None):
        """
        This method is used to visualize paths with details. Specifically, entity hierarchy for each entity will be
        printed.

        :param inputs:
        :param labels:
        :param type_weights:
        :param path_weights:
        :param rel:
        :param split:
        :param epoch:
        :param filter_negative_example:
        :param filter_false_prediction:
        :param probs:
        :param top_k_path:
        :param minimal_path_weight:
        :return:
        """

        num_ent_pairs, num_paths, num_steps, num_types = type_weights.shape
        highest_weighted_type_indices = np.argmax(type_weights, axis=3)

        rel_dir = os.path.join(self.save_dir, rel)
        if not os.path.exists(rel_dir):
            os.mkdir(rel_dir)
        rel_split_dir = os.path.join(rel_dir, split)
        if not os.path.exists(rel_split_dir):
            os.mkdir(rel_split_dir)
        file_name = os.path.join(rel_split_dir, str(epoch) + ".detailed.tsv")

        with open(file_name, "a") as fh:
            for ent_pairs_idx in range(num_ent_pairs):
                paths = []
                subj = None
                obj = None
                label = labels[ent_pairs_idx]

                # filter out negative examples
                if filter_negative_example:
                    if label == 0:
                        continue

                # filter out wrong predictions
                if filter_false_prediction:
                    if probs is not None:
                        prob = probs[ent_pairs_idx]
                        if abs(prob - label) > 0.5:
                            continue

                for path_idx in range(num_paths):
                    # Each path string should be: ent1[type1:weight1,...,typeC:weightC] - rel1 - ent2[type1:weight1,...,typeC:weightC]

                    # filter by path weight
                    if minimal_path_weight is not None and 0 < minimal_path_weight < 1:
                        if path_weights[ent_pairs_idx, path_idx] < minimal_path_weight:
                            continue

                    # processing a path
                    path = []
                    start = False
                    for stp in range(num_steps):
                        feats = inputs[ent_pairs_idx, path_idx, stp]
                        entity = feats[-2]
                        entity_name = self.idx2entity[entity]

                        # use dict to map freebase mid to name
                        if self.mid2name is not None:
                            if entity_name != "#PAD_TOKEN":
                                entity_name = entity_name.split(":")[1]
                            if entity_name in self.mid2name:
                                entity_name = self.mid2name[entity_name]

                        # ignore pre-paddings
                        if not start:
                            if entity_name != "#PAD_TOKEN":
                                start = True
                                if subj is None:
                                    subj = entity_name
                                else:
                                    assert subj == entity_name
                        if start:
                            rel = feats[-1]
                            types = feats[0:-2]
                            weights = type_weights[ent_pairs_idx, path_idx, stp]
                            types_str = []
                            for i in range(len(types)):
                                type_name = self.idx2entity_type[types[i]]
                                weight = weights[i]
                                type_str = type_name + ":" + "%.3f" % weight
                                types_str.append(type_str)
                            types_str = "[" + ",".join(types_str) + "]"
                            rel_name = self.idx2relation[rel]
                            path += [entity_name + types_str]
                            if rel_name != "#END_RELATION":
                                path += [rel_name]
                            if stp == num_steps - 1:
                                if obj is None:
                                    obj = entity_name
                                else:
                                    assert obj == entity_name
                    path_str = "-".join(path)
                    paths.append((path_str, path_weights[ent_pairs_idx, path_idx]))

                if not paths:
                    continue

                paths = sorted(paths, key=lambda x: x[1], reverse=True)
                # keep only top K paths
                if top_k_path is not None and top_k_path > 0:
                    paths = paths[0:min(len(paths), top_k_path)-1]

                weighted_paths = [p[0] + "," + str(p[1]) for p in paths]
                paths_str = " -#- ".join(weighted_paths)
                fh.write(subj + "," + obj + "\t" + str(label) + "\t" + paths_str + "\n")

    def visualize_paths_with_relation_and_type(self, inputs, labels, type_weights, path_weights, rel, split, epoch,
                                               filter_negative_example=False, filter_false_prediction=False, probs=None,
                                               top_k_path=None, minimal_path_weight=None):
        """
        This method is used to visualize paths in a compact way. Specifically, only the highest weighted entity type
        for each entity will be printed.

        :param inputs:
        :param labels:
        :param type_weights:
        :param path_weights:
        :param rel:
        :param split:
        :param epoch:
        :param filter_negative_example:
        :param filter_false_prediction:
        :param probs:
        :param top_k_path:
        :param minimal_path_weight:
        :return:
        """
        num_ent_pairs, num_paths, num_steps, num_types = type_weights.shape
        highest_weighted_type_indices = np.argmax(type_weights, axis=3)

        rel_dir = os.path.join(self.save_dir, rel)
        if not os.path.exists(rel_dir):
            os.mkdir(rel_dir)
        rel_split_dir = os.path.join(rel_dir, split)
        if not os.path.exists(rel_split_dir):
            os.mkdir(rel_split_dir)
        file_name = os.path.join(rel_split_dir, str(epoch) + ".tsv")

        with open(file_name, "a") as fh:
            for ent_pairs_idx in range(num_ent_pairs):
                paths = []
                subj = None
                obj = None
                label = labels[ent_pairs_idx]

                # filter out negative examples
                if filter_negative_example:
                    if label == 0:
                        continue

                # filter out wrong predictions
                if filter_false_prediction:
                    if probs is not None:
                        prob = probs[ent_pairs_idx]
                        if abs(prob - label) > 0.5:
                            continue

                for path_idx in range(num_paths):
                    # Each path string should be: type1 - rel1 - type2

                    # filter by path weight
                    if minimal_path_weight is not None and 0 < minimal_path_weight < 1:
                        if path_weights[ent_pairs_idx, path_idx] < minimal_path_weight:
                            continue

                    # processing a path
                    path = []
                    start = False
                    for stp in range(num_steps):
                        feats = inputs[ent_pairs_idx, path_idx, stp]
                        entity = feats[-2]
                        entity_name = self.idx2entity[entity]

                        # use dict to map freebase mid to name
                        if self.mid2name is not None:
                            if entity_name != "#PAD_TOKEN":
                                entity_name = entity_name.split(":")[1]
                            if entity_name in self.mid2name:
                                entity_name = self.mid2name[entity_name]

                        # ignore pre-paddings
                        if not start:
                            if entity_name != "#PAD_TOKEN":
                                start = True
                                if subj is None:
                                    subj = entity_name
                                else:
                                    assert subj == entity_name

                        if start:
                            rel = feats[-1]
                            types = feats[0:-2]
                            rel_name = self.idx2relation[rel]
                            highest_weighted_type = types[highest_weighted_type_indices[ent_pairs_idx, path_idx, stp]]
                            type_name = self.idx2entity_type[highest_weighted_type]
                            path += [type_name]
                            if rel_name != "#END_RELATION":
                                path += [rel_name]
                            if stp == num_steps - 1:
                                if obj is None:
                                    obj = entity_name
                                else:
                                    assert obj == entity_name
                    path_str = "-".join(path)
                    paths.append((path_str, path_weights[ent_pairs_idx, path_idx]))

                if not paths:
                    continue

                paths = sorted(paths, key=lambda x: x[1], reverse=True)
                # keep only top K paths
                if top_k_path is not None and top_k_path > 0:
                    paths = paths[0:min(len(paths), top_k_path)-1]
                weighted_paths = [p[0] + "," + str(p[1]) for p in paths]
                paths_str = " -#- ".join(weighted_paths)
                fh.write(subj + "," + obj + "\t" + str(label) + "\t" + paths_str + "\n")

    def visualize_contradictions(self, inputs, labels, type_weights, path_weights, relation, split,
                                 filter_false_prediction=False, probs=None, minimal_path_weight=None):
        """
        This method is used to extract contradiction examples. Another method needs to be called to print these examples

        :param inputs:
        :param labels:
        :param type_weights:
        :param path_weights:
        :param relation:
        :param split:
        :param filter_false_prediction:
        :param probs:
        :param minimal_path_weight:
        :return:
        """

        num_ent_pairs, num_paths, num_steps, num_types = type_weights.shape
        highest_weighted_type_indices = np.argmax(type_weights, axis=3)

        if split != "test":
            print("Skip generation of contradictions for split other than test")
            return

        if relation not in self.rel_path2contradictions:
            self.rel_path2contradictions[relation] = {}

        for ent_pairs_idx in range(num_ent_pairs):
            subj = None
            obj = None
            label = labels[ent_pairs_idx]

            # filter out wrong predictions
            if filter_false_prediction:
                if probs is not None:
                    prob = probs[ent_pairs_idx]
                    if abs(prob - label) > 0.5:
                        continue

            for path_idx in range(num_paths):

                # filter by path weight
                if minimal_path_weight is not None and 0 < minimal_path_weight < 1:
                    if path_weights[ent_pairs_idx, path_idx] < minimal_path_weight:
                        continue

                # processing a path
                path = []
                rel_path = []
                start = False
                for stp in range(num_steps):
                    feats = inputs[ent_pairs_idx, path_idx, stp]
                    entity = feats[-2]
                    entity_name = self.idx2entity[entity]

                    # use dict to map freebase mid to name
                    if self.mid2name is not None:
                        if entity_name != "#PAD_TOKEN":
                            entity_name = entity_name.split(":")[1]
                        if entity_name in self.mid2name:
                            entity_name = self.mid2name[entity_name]

                    # ignore pre-paddings
                    if not start:
                        if entity_name != "#PAD_TOKEN":
                            start = True
                            if subj is None:
                                subj = entity_name
                            else:
                                assert subj == entity_name

                    if start:
                        rel = feats[-1]
                        types = feats[0:-2]
                        rel_name = self.idx2relation[rel]
                        highest_weighted_type = types[highest_weighted_type_indices[ent_pairs_idx, path_idx, stp]]
                        type_name = self.idx2entity_type[highest_weighted_type]
                        path += [entity_name + "[" + type_name + "]"]
                        if rel_name != "#END_RELATION":
                            path += [rel_name]
                            rel_path += [rel_name]
                        if stp == num_steps - 1:
                            if obj is None:
                                obj = entity_name
                            else:
                                assert obj == entity_name
                path_str = "-".join(path)
                rel_path_str = "-".join(rel_path)

                if rel_path_str not in self.rel_path2contradictions[relation]:
                    self.rel_path2contradictions[relation][rel_path_str] = []
                # each example will be (subj, obj, label): weight, subj[type1]-ent2[type2]-obj[type3]
                example_str = "(" + subj + ", " + obj + ", " + str(label) + "): " + str(path_weights[ent_pairs_idx, path_idx]) + ", " + path_str
                if label == 0:
                    self.rel_path2contradictions[relation][rel_path_str].append(example_str)
                else:
                    self.rel_path2contradictions[relation][rel_path_str].insert(0, example_str)

    def print_contradictions(self, rel):
        """
        This method is used to write contradiction examples.

        :param rel:
        :return:
        """

        if rel not in self.rel_path2contradictions:
            print("Relation {} does not have any contradictory examples".format(rel))
            return

        rel_dir = os.path.join(self.save_dir, rel)
        if not os.path.exists(rel_dir):
            os.mkdir(rel_dir)
        rel_split_dir = os.path.join(rel_dir, "test")
        if not os.path.exists(rel_split_dir):
            os.mkdir(rel_split_dir)
        file_name = os.path.join(rel_split_dir, "contradictions.tsv")

        with open(file_name, "a") as fh:
            for idx, rel_path in enumerate(self.rel_path2contradictions[rel]):
                for example in self.rel_path2contradictions[rel][rel_path]:
                    fh.write(str(idx) + "\t" + rel_path + "\t" + example + "\n")

    def save_space(self, rel, best_epoch):
        """
        This method is used to delete visualizations that are not from the best models in order to save disk space.

        :param rel:
        :param best_epoch:
        :return:
        """
        rel_dir = os.path.join(self.save_dir, rel)
        for split in os.listdir(rel_dir):
            rel_split_dir = os.path.join(rel_dir, split)
            for file_name in os.listdir(rel_split_dir):
                epoch = int(file_name.split(".")[0])
                if epoch == 0 or epoch == best_epoch or epoch == 29:
                    continue
                # print(file_name)
                os.remove(os.path.join(rel_split_dir, file_name))