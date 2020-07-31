import collections
import time
import itertools
# from CythonPathsExtractor.CythonPathsExtractor import PythonPathsExtractor
import os
import pickle
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

"""
This class extracts paths between all entity pairs in split.
Using Cython reduce 1/3 of time.
"""
class PathExtractor:
    """
    This class uses Bidirectional Breadth First Search (BFS) to find paths between entities.

    :ivar max_length: the maximum number of relations in an extracted paths
    :ivar include_entity: a path can be a sequence of relations if include_entity is set to False or a sequence of
                          alternating relation and entities if include_entity is True.
    :ivar include_path_len1: whether extract paths with length equal to 1
    :ivar max_paths_per_pair: Default None. When max_paths_per_pair is set to a real value, if the number of paths between an entity
                              pair exceeds the value, we will sample the number of paths equal to this value from all
                              paths.
    :ivar multiple_instances_per_pair: Default False. When we want to augment the data, we can include multiple instances
                                       of an entity pair, each instance with different subset of all paths between this
                                       entity pair.
    :ivar max_instances_per_pair: Default None.
    :ivar paths_sample_method: Default "random". If the method is set to "all_lengths", paths with each length will be
                               sampled separately in order to ensure diversity. This is necessary because the number of
                               longer paths will be far more than that of shorter paths before sampling.
    :ivar relation_to_pairs_to_paths:
    :ivar relation_to_path_types:

    .. note::

        Using Cython reduce 1/3 of time.
    """

    def __init__(self, max_length, include_entity, save_dir, include_path_len1,
                 max_paths_per_pair=None, multiple_instances_per_pair=False, max_instances_per_pair=None,
                 paths_sample_method="random"):
        """
        :param max_length:
        :param include_entity:
        :param save_dir:
        :param include_path_len1:
        :param max_paths_per_pair:
        :param multiple_instances_per_pair:
        :param max_instances_per_pair:
        :param paths_sample_method:
        """
        self.max_length = max_length
        if self.max_length <= 1:
            raise Exception("Max path length needs to be greater than 1.")
        if self.max_length % 2 != 0:
            print("Because we are using Bidirectional BFS, it's better to have even number max length.")
        self.include_entity = include_entity
        self.include_path_len1 = include_path_len1
        self.ignore_no_path_entity_pair = False

        self.max_paths_per_pair = max_paths_per_pair
        self.multiple_instances_per_pair = multiple_instances_per_pair
        self.max_instances_per_pair = max_instances_per_pair
        self.paths_sample_method = paths_sample_method

        # Create directory to save extracted paths
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # {rel: [path1,..,pathN]} when multiple_instances_per_pair is False
        # {rel: [[path1,..,pathN],..,[pathP,..,pathQ]]}, O.W.
        self.relation_to_pairs_to_paths = {}
        # {rel: set(path_types)}
        self.relation_to_path_types = {}

    def extract_paths(self, graph, split, vocabs):
        """
        This function is used to extract paths between entity pairs in the given split from the given graph

        :param graph: :meth:`main.graphs.AdjacencyGraph`
        :param split: :meth:`main.data.Split`
        :param vocabs: :meth:`main.data.Vocabs`
        :return:
        """
        for rel in split.relation_to_splits_to_instances:
            self.relation_to_path_types[rel] = set()
            self.relation_to_pairs_to_paths[rel] = {}
            start_time = time.time()
            for spt in split.relation_to_splits_to_instances[rel]:
                print("Extract", spt, "paths for relation:", rel)
                for subj, obj, label in tqdm(split.relation_to_splits_to_instances[rel][spt]):
                    # st= time.time()
                    paths_dict = self.get_paths(subj, rel, obj, graph, vocabs)
                    # Returned paths is a dictionary mapping from path length to a set of path strings
                    paths = []
                    for length in paths_dict:
                        paths += [p for p in paths_dict[length]]
                    #print(subj, obj, "has", len(paths), "paths")

                    if not self.multiple_instances_per_pair:
                        if self.max_paths_per_pair is not None:
                            if len(paths) > self.max_paths_per_pair:
                                if self.paths_sample_method == "random":
                                    choices = np.random.choice(len(paths), self.max_paths_per_pair, replace=False)
                                    selected_paths = [paths[i] for i in choices]
                                elif self.paths_sample_method == "all_lengths":
                                    num_path_lengths = len(paths_dict)
                                    num_paths_per_length = int(self.max_paths_per_pair / num_path_lengths)
                                    selected_paths = []
                                    for path_length in paths_dict:
                                        if len(paths_dict[path_length]) < num_paths_per_length:
                                            selected_paths += list(paths_dict[path_length])
                                        else:
                                            selected_paths += list(np.random.choice(list(paths_dict[path_length]), num_paths_per_length, replace=False))
                            else:
                                selected_paths = paths
                    else:
                        assert self.max_instances_per_pair is not None
                        paths_list = []
                        if len(paths) < self.max_paths_per_pair:
                            paths_list.append([paths])
                        else:
                            num_instances = min(self.max_instances_per_pair, int(len(paths)/self.max_paths_per_pair))
                            for i in range(num_instances):
                                paths_list.append([paths[c] for c in np.random.choice(len(paths), self.max_paths_per_pair, replace=False)])

                    # print(subj, obj, "has", len(paths), "paths", time.time() - st) #, "paths:", paths)
                    if not self.multiple_instances_per_pair:
                        self.relation_to_pairs_to_paths[rel][(subj, obj)] = selected_paths
                        self.relation_to_path_types[rel].update(selected_paths)
                    else:
                        self.relation_to_pairs_to_paths[rel][(subj, obj)] = paths_list
                        for paths in paths_list:
                            self.relation_to_path_types[rel].update(paths)

            end_time = time.time()
            print("Takes", end_time - start_time)

    def write_paths(self, split):
        """
        This function write extracted paths to files.

        :param split: :meth:`main.data.Split`
        :return:
        """
        params = {"simple": True, "max_length": self.max_length, "include_entity": self.include_entity,
                  "include_path_len1": self.include_path_len1,
                  "ignore_no_path_entity_pair": self.ignore_no_path_entity_pair,
                  "max_paths_per_pair": self.max_paths_per_pair,
                  "multiple_instances_per_pair": self.multiple_instances_per_pair,
                  "max_instances_per_pair": self.max_instances_per_pair}

        with open(os.path.join(self.save_dir, "params.json"), "w+") as fh:
            json.dump(params, fh)

        for rel in split.relation_to_splits_to_instances:
            rel_dir = os.path.join(self.save_dir, rel)
            if not os.path.exists(rel_dir):
                os.mkdir(rel_dir)
            for spt in split.relation_to_splits_to_instances[rel]:
                spt_filename = os.path.join(rel_dir, spt + "_matrix.tsv")
                with open(spt_filename, "w+") as fh:
                    for subj, obj, label in split.relation_to_splits_to_instances[rel][spt]:
                        assert (subj, obj) in self.relation_to_pairs_to_paths[rel]
                        if not self.multiple_instances_per_pair:
                            paths = self.relation_to_pairs_to_paths[rel][(subj, obj)]
                            paths_string = " -#- ".join(paths)
                            fh.write(subj + "," + obj + "\t" + str(label) + "\t" + paths_string + "\n")
                        else:
                            for paths in self.relation_to_pairs_to_paths[rel][(subj, obj)]:
                                paths_string = " -#- ".join(paths)
                                fh.write(subj + "," + obj + "\t" + str(label) + "\t" + paths_string + "\n")

    def get_paths(self, source, target_relation, target, graph, vocabs):
        """
        This function finds paths between two entities. This function performs bi-directional BFS by calling two BFS
        searches from the source and the target entities.

        :param source: source entity
        :param target_relation: the target relation that paths between the two entities are used to predict. Extracted
                                paths between the two entities will exclude this target relation
        :param target: target entity
        :param graph: :meth:`main.graphs.AdjacencyGraph`
        :param vocabs: :meth:`main.data.Vocabs`
        :return:
        """
        source_idx = vocabs.node_to_idx[source]
        target_idx = vocabs.node_to_idx[target]
        target_relation_idx = vocabs.relation_to_idx[target_relation]

        # paths_dict is a dictionary mapping from path length to a set of path strings
        paths_dict = {}
        if source_idx == target_idx:
            return paths_dict
        if source_idx not in graph.node_to_parents and source_idx not in graph.node_to_children:
            return paths_dict
        if target_idx not in graph.node_to_parents and target_idx not in graph.node_to_children:
            return paths_dict

        source_subgraph = self.bfs_from_node(source_idx, target_relation_idx, target_idx, graph, vocabs, self.max_length / 2)
        target_subgraph = self.bfs_from_node(target_idx, vocabs.idx_to_rev_relation_idx[target_relation_idx], source_idx, graph, vocabs, self.max_length / 2)
        # print(len(source_subgraph))
        # print(len(target_subgraph))
        # print(source_subgraph)
        # print(target_subgraph)

        # combine subgraphs
        # situation 1
        if target_idx in source_subgraph:
            paths_from_source = source_subgraph[target_idx]
            for path in paths_from_source:
                if not self.include_path_len1:
                    assert len(path) > 3
                path_str, path_len = self.path_to_string(path, vocabs)
                if path_len not in paths_dict:
                    paths_dict[path_len] = set()
                paths_dict[path_len].add(path_str)
        if source_idx in target_subgraph:
            paths_from_target = target_subgraph[source_idx]
            for path in paths_from_target:
                if not self.include_path_len1:
                    assert len(path) > 3
                path_str, path_len = self.path_to_string(path, vocabs, reverse=True)
                if path_len not in paths_dict:
                    paths_dict[path_len] = set()
                paths_dict[path_len].add(path_str)
        # situation 2
        intersections = set(source_subgraph.keys()).intersection(set(target_subgraph.keys()))
        # print("intersections", intersections)
        for common_node_idx in intersections:
            source_to_common_node_paths = source_subgraph[common_node_idx]
            target_to_common_node_paths = target_subgraph[common_node_idx]
            # print("source paths to node", len(source_to_common_node_paths), "target paths to node", len(target_to_common_node_paths))
            source_paths = set([(self.path_to_string(source_to_common_node_path, vocabs, drop_last=True)) for source_to_common_node_path in source_to_common_node_paths])
            target_paths = set([(self.path_to_string(target_to_common_node_path, vocabs, reverse=True)) for target_to_common_node_path in target_to_common_node_paths])
            # print(source_paths)
            # print(target_paths)

            for source_path in source_paths:
                for target_path in target_paths:
                    # add to set maybe a bottleneck
                    if self.include_entity:
                        path_str = source_path[0] + target_path[0]
                        path_len = source_path[1] + target_path[1]
                        if path_len not in paths_dict:
                            paths_dict[path_len] = set()
                        paths_dict[path_len].add(path_str)
                    else:
                        path_str = source_path[0] + "-" + target_path[0]
                        path_len = source_path[1] + target_path[1]
                        if path_len not in paths_dict:
                            paths_dict[path_len] = set()
                        paths_dict[path_len].add(path_str)
        return paths_dict

    def bfs_from_node(self, source, target_relation, target, graph, vocabs, steps):
        """
        This function uses BFS to find paths between two entities. All entities, relations, and graph use indices.

        :param source: source entity
        :param target_relation: the target relation that paths between the two entities are used to predict. Extracted
                                paths between the two entities will exclude this target relation
        :param target: target entity
        :param graph: :meth:`main.graphs.AdjacencyGraph`
        :param vocabs: :meth:`main.data.Vocabs`
        :param steps: max depths of the search
        :return:

        .. note::

            If the real path in the graph is source -> edge1 -> entity1 -> edge2 -> target, the path will be a
            Tuple(source, edge1, entity1, edge2, target)
        """
        # double ended queue. use append() and popleft() for FIFO.
        queue = collections.deque()
        queue.append((source, tuple([source]), steps))
        # subgraph is {end node:{path types}}
        subgraph = {}
        while queue:
            cur_node, path_so_far, steps_left = queue.popleft()
            if len(path_so_far) > 1:
                if cur_node not in subgraph:
                    subgraph[cur_node] = set()
                subgraph[cur_node].add(path_so_far)
            if steps_left > 0:
                if cur_node in graph.node_to_parents:
                    parents = graph.node_to_parents[cur_node]
                    # print(vocabs.idx_to_node[cur_node], "has parents", [vocabs.idx_to_node[p] for p in parents])
                else:
                    parents = set()
                if cur_node in graph.node_to_children:
                    children = graph.node_to_children[cur_node]
                    # print(vocabs.idx_to_node[cur_node], "has children", [vocabs.idx_to_node[c] for c in children])
                else:
                    children = set()

                # debug: max fanout
                #if len(children) + len(parents) > 100:
                #    continue

                for neighbor in parents | children:
                    # loop is detected here. only check neighbor against entity node in the path. This is neccessary bc
                    # relation and entity could share the same index.
                    if neighbor in path_so_far[::2]:
                        continue

                    edges = graph.pair_to_relations[(cur_node, neighbor)]
                    for edge in edges:
                        # Important: We need to make sure the target relation is ignored
                        if self.include_path_len1:
                            if cur_node == source and neighbor == target:
                                if edge == target_relation:
                                    continue
                            if cur_node == target and neighbor == source:
                                if edge == vocabs.idx_to_rev_relation_idx[target_relation]:
                                    continue

                        if neighbor == target:
                            # condition below works for both when entities are included and not included
                            # included: source -> edge1 -> entity1
                            # not included: edge1
                            if self.include_path_len1 or len(path_so_far) > 1:
                                if neighbor not in subgraph:
                                    subgraph[neighbor] = set()
                                subgraph[neighbor].add(path_so_far + (edge, neighbor))
                        else:
                            queue.append((neighbor, path_so_far + (edge, neighbor), steps_left - 1))
        return subgraph

    def path_to_string(self, path, vocabs, reverse=False, drop_last=False):
        """
        This function formats the path with entity indices and relation indices to its string with entity names and
        relation names. It also count the number of relations in the path.

        :param path: Tuple(entity1, edge1, entity2, edge2, entity3)
        :param vocabs: :meth:`main.data.Vocabs`
        :param reverse: if reversed, the string will be "entity3#rev_edge2#entity2#rev_edge1#entity1"
        :param drop_last: if self.include_entity and drop_last, the string will be "entity1-edge1-entity2-edge2-"
        :return: Tuple("entity1-edge1-entity2-edge2-entity3", length counting the number of relations)
        """
        path_string = ""
        if self.include_entity:
            if not reverse:
                for idx in range(0, (len(path)-1)/2):
                    path_string += vocabs.idx_to_node[path[idx*2]]
                    path_string += "-" + vocabs.idx_to_relation[path[idx*2+1]] + "-"
                    if not drop_last:
                        if idx == (len(path)-1)/2 - 1:
                            path_string += vocabs.idx_to_node[path[idx*2+2]]
            else:
                for idx in range((len(path)-1)/2, 0, -1):
                    path_string += vocabs.idx_to_node[path[idx*2]]
                    path_string += "-" + vocabs.idx_to_relation[vocabs.idx_to_rev_relation_idx[path[idx*2-1]]] + "-"
                    if idx == 1:
                        path_string += vocabs.idx_to_node[path[0]]
        else:
            if not reverse:
                for idx in range(0, (len(path)-1)/2):
                    if idx == int((len(path)-1)/2) - 1:
                        path_string += vocabs.idx_to_relation[path[idx*2+1]]
                    else:
                        path_string += vocabs.idx_to_relation[path[idx*2+1]] + "-"

            else:
                for idx in range(int((len(path)-1)/2), 0, -1):
                    if idx == 1:
                        path_string += vocabs.idx_to_relation[vocabs.idx_to_rev_relation_idx[path[idx*2-1]]]
                    else:
                        path_string += vocabs.idx_to_relation[vocabs.idx_to_rev_relation_idx[path[idx*2-1]]] + "-"
        return path_string, int((len(path)-1)/2)


