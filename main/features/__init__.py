class ExtractingPaths:

    def __init__(self, relation_instances, graph, node_to_neighbors, pair_to_relations, node_to_idx, relation_to_idx, idx_to_node, idx_to_relation, maximum_length=10, include_entity=False, multigraph=True, verbose=True, save_dir='./data/paths', save_to_pickle=False, save_to_txt=False, remaining_percentage=1.0, random_seed=1):
        """
        :param relation_instances: relation_instances is a dict (keyed by relation names) of dict (keyed by subject) of dict (keyed by object).
                                   {rel: {subj: {obj: True/False/None}}}
                                   The inner-most dict stores the value of the relation instance (True, False, None)
        :param maximum_length: the maximum allowed path length. A path is defined as a sequence of relations
        :param include_entity: whether the path should also include entities in addition to relations
        :param multigraph: whether more than one relation can exist between two entities
        :param verbose:
        :param save_dir: where to save extracted paths
        """
        # Init variables
        ############test################
        self.include_entity = True # include_entity
        self.verbose = verbose
        self.maximum_length = maximum_length
        self.multigraph = multigraph
        self.remaining_percentage = remaining_percentage
        self.random_seed = random_seed

        # Store graph
        self.graph = graph

        # Store relation instances
        self.relation_instances = relation_instances

        # Map node names to numbers and map relation names to numbers
        self.node_to_idx = node_to_idx
        self.relation_to_idx = relation_to_idx
        self.idx_to_node = idx_to_node
        self.idx_to_relation = idx_to_relation

        # Cache neighbors of each node and store relations between two nodes
        self.node_to_neighbors = node_to_neighbors
        self.pair_to_relations = pair_to_relations

        # Create directory to save extracted paths
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_to_pickle = save_to_pickle
        self.save_to_txt = save_to_txt

    def extract_paths(self, rel, file_dir):
        # Method 3 extractor
        # pe_ptr is a python interface for cython c++ extractor. This is actually slower than python extractor.
        # pe_ptr = PythonPathsExtractor(self.node_to_neighbors, self.maximum_length)

        if rel not in self.relation_instances:
            raise Exception("Relation type "+rel+" is not in the graph")

        print('Extract paths for entity pairs of relation:', rel)

        start_all = time.time()

        if self.save_to_pickle:
            all_paths_for_rel = {}
        for subj in self.relation_instances[rel]:
            for obj in self.relation_instances[rel][subj]:
                if str(subj) != "sandwich.n.01" or str(obj) != "mug.n.04":
                    continue
                # ignore synthetic edge
                if self.relation_instances[rel][subj][obj] == 4 or self.relation_instances[rel][subj][obj] == 5:
                    continue
                # if obj == subj:
                #     continue
                start = time.time()
                # Method 1. networkx built in function. This can only find directed paths
                # paths = list(nx.all_simple_paths(self.graph, subj, obj, self.maximum_length))

                # Method 2. custom
                # paths = self.get_all_simple_paths(self.node_to_idx[subj], self.node_to_idx[obj], cutoff=self.maximum_length)

                ##############test################
                paths = self.get_paths(self.node_to_idx[subj], self.node_to_idx[obj], rel, max_length=self.maximum_length)

                if self.save_to_txt:
                    self.write_and_filter_paths(subj, obj, rel, self.relation_instances[rel][subj][obj], paths)
                if self.save_to_pickle:
                    ##############test################
                    all_paths_for_rel[(subj, obj)] = paths # self.filter_paths(paths)
                    print(all_paths_for_rel[(subj, obj)])
                end = time.time()

                # if self.verbose:
                print("For (" + str(subj) + ", " + str(obj) + "), " + str(len(paths)) + " paths, takes " + str(end - start) + "s: ")

                # Method 3. Use cython c++ extractor. slower
                # paths = pe_ptr.get_paths(self.node_to_idx[subj], self.node_to_idx[obj])

            if self.save_to_pickle:
                pickle.dump(all_paths_for_rel, open(file_dir, "wb"))

        end_all = time.time()
        print(end_all - start_all)

    def extract_paths_for_subject_object_pair(self, subject, object):
        # Method 3 extractor
        # pe_ptr is a python interface for cython c++ extractor. This is actually slower than python extractor.
        # pe_ptr = PythonPathsExtractor(self.node_to_neighbors, self.maximum_length)

        start = time.time()
        # Method 1. networkx built in function. This can only find directed paths
        # paths = list(nx.all_simple_paths(self.graph, subj, obj, self.maximum_length))

        # Method 2. custom
        paths = self.get_all_simple_paths(self.node_to_idx[subject], self.node_to_idx[object], cutoff=self.maximum_length)
        path_instances = self.filter_paths(paths)
        end = time.time()
        if self.verbose:
            print("For (" + str(subject) + ", " + str(object) + "), " + str(len(path_instances)) + " paths, takes " + str(end - start) + "s: ")

        # Method 3. Use cython c++ extractor. slower
        # paths = pe_ptr.get_paths(self.node_to_idx[subj], self.node_to_idx[obj])
        return path_instances

    def get_all_simple_paths(self, source, target, cutoff=10):
        if cutoff < 1:
            return
        # 1. find path
        visited = [source]
        # this is a list of iterators
        if source not in self.node_to_neighbors:
            return []
        stack = [iter(self.node_to_neighbors[source])]
        paths = []
        while stack:
            children = stack[-1]
            child = next(children, -1)
            if child == -1:
                stack.pop()
                visited.pop()
            elif len(visited) < cutoff:
                if child == target:
                    paths.append(visited + [child])
                elif child not in visited:
                    if child not in self.node_to_neighbors:
                        continue
                    visited.append(child)
                    stack.append(iter(self.node_to_neighbors[child]))
            else:  # len(visited) == cutoff:
                if self.multigraph:
                    # IMPORTANT: this part is different between DiGraph and MultiDiGraph. For MultiDiGraph, we need to
                    # enumerate all children of current node because there may be more than one relation that connects
                    # current node to target node
                    if child == target:
                        paths.append(visited + [child])
                    for c in children:
                        if c == target:
                            paths.append(visited + [c])
                    # Important: No need to pop because children of current node will all be visited and in next
                    # iteration of the while loop, child will be None.
                    # stack.pop()
                    # visited.pop()
                else:
                    if child == target:
                        paths.append(visited + [child])
                        stack.pop()
                        visited.pop()
                    else:
                        for c in children:
                            if c == target:
                                paths.append(visited + [c])
                                stack.pop()
                                visited.pop()
        return self.expand_paths_by_nodes(paths)

    def expand_paths_by_nodes(self, paths):
        """
        This function takes in all paths that are represented as lists of consecutive nodes [node1, node2,...,nodeN]
        and converted to paths represented as
        lists of consecutive relations [rel1, rel2,...,relM] if self.include_entity is false, or as
        lists of nodes and relations [node1, rel1, node2, rel2,...,relM, nodeN] if self.include_entity is true.

        :param paths: a list of lists of node indexes
        :return: a list of the chosen path representations
        """
        paths_formatted = set()
        # Expand each path
        for path in paths:
            if len(path) < 2:
                continue
            expanded_paths = set()
            if self.include_entity:
                relations_for_each_step = [[path[0]]]
            else:
                relations_for_each_step = []
            for index in range(1, len(path)):
                node1 = path[index-1]
                node2 = path[index]
                if (node1, node2) in self.pair_to_relations:
                    relations = self.pair_to_relations[(node1, node2)]
                else:
                    print(node1, node2)
                relations_for_each_step.append(relations)
                if self.include_entity:
                    relations_for_each_step.append([node2])
            expanded_paths.update(list(itertools.product(*relations_for_each_step)))
            paths_formatted.update(expanded_paths)
        return paths_formatted

    def write_and_filter_paths(self, source, target, relation, label, paths):
        """
        This function is used to write all paths between any two entities that are connected by the input relation to a
        file. Because this function will go through all paths node by node, this function will also used to filter paths
        to save computation.

        :param source: name of the source node
        :param target: name of the target node
        :param relation: name of the relation between the two nodes
        :param label: label for this pair of nodes (whether the relation exist)
        :param paths: a list of path representations
        :return:
        """
        file_dir = os.path.join(self.save_dir, relation + "_" + str(self.maximum_length) + "_" + str(self.remaining_percentage) + "_" + str(self.random_seed) + ".txt")
        with open(file_dir, "a") as fh:
            fh.write(str(label) + "\t" + str(source) + "\t" + str(target) + "\t")
            for pdx, path in enumerate(paths):
                if not self.include_entity:
                    if len(path) == 1:
                        continue
                    for rdx, rel_idx in enumerate(path):
                        fh.write(self.idx_to_relation[rel_idx])
                        if rdx != len(path)-1:
                            fh.write("|")
                    if pdx != len(paths)-1:
                        fh.write("###")
                else:
                    if len(path) == 3:
                        continue
                    fh.write(self.idx_to_node[path[0]].get_name())
                    fh.write("|")
                    for rdx in range(0, (len(path)-1)/2):
                        fh.write(self.idx_to_relation[path[rdx*2+1]])
                        fh.write("|")
                        fh.write(self.idx_to_node[path[rdx*2+2]].get_name())
                        if rdx*2+2 != len(path)-1:
                            fh.write("|")
                    if pdx != len(paths)-1:
                        fh.write("###")
            fh.write("\n")

    def filter_paths(self, paths):
        """
        This function is used to filter all paths and change paths represented by relation index and entity index to
        paths represented by relation name and entity name
        :param paths: a set of tuples
        :return:
        """
        formatted_paths = set()
        for path in paths:
            formatted_path = []
            if self.include_entity:
                if len(path) == 3:
                    continue
                formatted_path.append(self.idx_to_node[path[0]].get_name())
                for rdx in range(0, (len(path)-1)/2):
                    formatted_path.append(self.idx_to_relation[path[rdx*2+1]])
                    formatted_path.append(self.idx_to_node[path[rdx*2+2]].get_name())
            else:
                if len(path) == 1:
                    continue
                for rel_idx in path:
                    formatted_path.append(self.idx_to_relation[rel_idx])
            formatted_paths.add(tuple(formatted_path))
        return formatted_paths


    #####################################test########################################

    def path_to_string(self, path):
        str = ""
        str += (self.idx_to_node[path[0]].get_name())
        for rdx in range(0, (len(path)-1)/2):
            str += "#" + (self.idx_to_relation[path[rdx*2+1]])
            str += "#" + (self.idx_to_node[path[rdx*2+2]].get_name())
        return str

    def rev_path_to_string(self, path):
        str = ""
        str += (self.idx_to_node[path[0]].get_name())
        for rdx in range(0, (len(path)-1)/2):
            str += "#" + (self.idx_to_relation[self.idx_to_rev_relation_idx[path[rdx*2+1]]])
            str += "#" + (self.idx_to_node[path[rdx*2+2]].get_name())
        return str

    def path_to_string_no_last_entity(self, path):
        str = ""
        for rdx in range(0, len(path)/2):
            if rdx == 0:
                str += (self.idx_to_node[path[rdx*2]].get_name())
            else:
                str += "#" + (self.idx_to_node[path[rdx*2]].get_name())
            str += "#" + (self.idx_to_relation[path[rdx*2+1]])
        return str

    def get_paths(self, source, target, relation, max_length=10):

        # improvement: generate when caching graph
        self.idx_to_rev_relation_idx = {}
        for idx in self.idx_to_relation:
            rel = self.idx_to_relation[idx]
            for idx2 in self.idx_to_relation:
                rel2 = self.idx_to_relation[idx2]
                if "-" in rel:
                    if "-" in rel2:
                        continue
                    else:
                        if rel.replace("-", "") == rel2:
                            self.idx_to_rev_relation_idx[idx] = idx2
                            self.idx_to_rev_relation_idx[idx2] = idx
                else:
                    if "-" in rel2:
                        if rel.replace("-", "") == rel2:
                            self.idx_to_rev_relation_idx[idx] = idx2
                            self.idx_to_rev_relation_idx[idx2] = idx
                    else:
                        continue

        for idx in self.idx_to_rev_relation_idx:
            rev_idx = self.idx_to_rev_relation_idx[idx]
            # print(self.idx_to_relation[idx], self.idx_to_relation[rev_idx])

        paths = set()
        if source == target:
            return paths
        if max_length <= 1:
            return paths
        if source not in self.node_to_neighbors:
            return paths
        if target not in self.node_to_neighbors:
            return paths
        # print(source, target, relation)

        source_subgraph = self.bfs_from_node(source, target, relation, max_length / 2)
        target_subgraph = self.bfs_from_node(target, source, relation, max_length / 2)
        # print(len(source_subgraph))
        # print(len(target_subgraph))

        # combine subgraphs
        # situation 1
        if target in source_subgraph:
            paths_from_source = source_subgraph[target]
            for path in paths_from_source:
                # improvement: this only works if include entity in path
                if len(path) > 3:
                    paths.add(self.path_to_string(path))
        if source in target_subgraph:
            paths_from_target = target_subgraph[source]
            for path in paths_from_target:
                if len(path) > 3:
                    paths.add(self.rev_path_to_string(path[::-1]))
        # situation 2
        intersections = set(source_subgraph.keys()).intersection(set(target_subgraph.keys()))
        # print("intersections", len(intersections))
        for common_node in intersections:
            source_to_common_node_paths = source_subgraph[common_node]
            target_to_common_node_paths = target_subgraph[common_node]
            # print("source paths to node", len(source_to_common_node_paths))
            # print("target paths to node", len(target_to_common_node_paths))

            # improvement: maybe the bfs_from_node can return path in different format. if walking from source or walking from target.
            # then we don't need to go through every path to take out the last element or reverse the order.

            source_paths = set([self.path_to_string_no_last_entity(source_to_common_node_path[0:-1]) for source_to_common_node_path in source_to_common_node_paths])
            target_paths = set([self.rev_path_to_string(target_to_common_node_path[::-1]) for target_to_common_node_path in target_to_common_node_paths])

            for source_path in source_paths:
                for target_path in target_paths:
                    # add to set maybe a bottleneck
                    paths.add(source_path + "#" + target_path)
        return paths

    # improvement: use idx to distinguish which node is using idx representation
    def bfs_from_node(self, source, target, relation, steps):

        # double ended queue. use append() and popleft() for FIFO.
        queue = collections.deque()
        queue.append((source, tuple([source]), steps))
        # map of (end node -> path types)
        subgraph = {}
        while queue:
            cur_node, path_to_node, steps_left = queue.popleft()
            if len(path_to_node) > 1:
                if cur_node not in subgraph:
                    subgraph[cur_node] = set()
                subgraph[cur_node].add(path_to_node)
            if steps_left > 0:
                neighbors = self.node_to_neighbors[cur_node]
                print(self.idx_to_node[cur_node], "has neighbors", len(neighbors))
                for neighbor in neighbors:
                    # loop is detected here
                    if neighbor in path_to_node:
                        continue
                    potential_skip = False
                    if (cur_node == source and neighbor == target) or (cur_node == target and neighbor == source):
                        potential_skip = True
                    relations_between = self.pair_to_relations[(cur_node, neighbor)]
                    for rel in relations_between:
                        # should skip this rel because this needs to be predicted
                        if potential_skip:
                            # improvement: this may not be efficient
                            # if self.relation_to_idx[relation] == rel or self.relation_to_idx[relation] == self.idx_to_rev_relation_idx[rel]:
                            if self.idx_to_relation[rel].replace("-", "") == relation.replace("-", ""):
                               continue

                        print(self.idx_to_node[cur_node], self.idx_to_relation[rel],  self.idx_to_node[neighbor])

                        if neighbor == target:
                            if neighbor not in subgraph:
                                subgraph[neighbor] = set()
                            subgraph[neighbor].add(path_to_node + (rel, neighbor))
                        else:
                            queue.append((neighbor, path_to_node + (rel, neighbor), steps_left - 1))
        # print(subgraph)
        return subgraph