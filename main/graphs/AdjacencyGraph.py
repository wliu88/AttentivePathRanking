# Improvement: using string as dictionary keys maybe faster.


class AdjacencyGraph:
    """
    This class manages data split (train/test/dev).

    :ivar node_to_parents: a dict from a node to a set of it's parents. Nodes are node ids.
    :ivar node_to_children: a dict from a node to a set of all it's children. Nodes are node ids
    :ivar pair_to_relations: a dict from a pair of nodes to a set of connecting relations.
                            (There may be multiple relations between two nodes).
                            Nodes are node ids. Relations are relation ids.
                            The order of the two entities in the pair matters since the graph is directed

    .. note::

        All positive relation instances from train, dev, and test are used to construct the graph. This is fine because test
        relation instances only need to not include the relation or its equivalent relations as length 1 paths.
    """
    def __init__(self):
        self.pair_to_relations = {}
        self.node_to_parents = {}
        self.node_to_children = {}

    def build_graph(self, typed_relation_instances, vocabs):
        """
        This function pre-computes neighbors of all nodes.

        :param typed_relation_instances: :meth:`main.data.TypedRelationInstances`
        :param vocabs: :meth:`main.data.Vocabs`
        :return:
        """
        for rel in typed_relation_instances.relation_to_instances:
            for subj, obj, label in typed_relation_instances.relation_to_instances[rel]:
                if label == 1:
                    # forward direction source ->edge-> target
                    source = vocabs.node_to_idx[subj]
                    target = vocabs.node_to_idx[obj]
                    edge = vocabs.relation_to_idx[rel]
                    if source not in self.node_to_children:
                        self.node_to_children[source] = set()
                    self.node_to_children[source].add(target)
                    if target not in self.node_to_parents:
                        self.node_to_parents[target] = set()
                    self.node_to_parents[target].add(source)
                    if (source, target) not in self.pair_to_relations:
                        self.pair_to_relations[(source, target)] = set()
                    self.pair_to_relations[(source, target)].add(edge)

                    # reverse direction target ->rev_edge-> source
                    rev_edge = vocabs.relation_to_idx["_" + rel]
                    if (target, source) not in self.pair_to_relations:
                        self.pair_to_relations[(target, source)] = set()
                    self.pair_to_relations[(target, source)].add(rev_edge)