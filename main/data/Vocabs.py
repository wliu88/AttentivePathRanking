class Vocabs:
    """This class manages the vocabularies of all entites and relations.

    :ivar idx_to_node: a dict mapping from an entity index to an entity
    :ivar node_to_idx: a dict mapping from an entity to an entity index
    :ivar idx_to_relation: a dict mapping from a relation index to a relation
    :ivar relation_to_idx: a dict mapping from a relation to a relation index
    :ivar idx_to_rev_relation_idx: a dict mapping from the index of a relation to the index of its reverse relation.
    """

    def __init__(self):
        self.idx_to_node = {}
        self.node_to_idx = {}
        self.idx_to_relation = {}
        self.relation_to_idx = {}
        self.idx_to_rev_relation_idx = {}

    def build_vocabs(self, typed_relation_instances):
        """This function builds vocabularies of relations and entities from relation instances.

        :param typed_relation_instances: all relation instances
        :type typed_relation_instances: :meth:`main.data.TypedRelationInstances`
        """
        for rel in typed_relation_instances.relation_to_instances:
            if rel not in self.relation_to_idx:
                self.relation_to_idx[rel] = len(self.relation_to_idx)
                self.idx_to_relation[self.relation_to_idx[rel]] = rel
                rev_rel = "_" + rel
                self.relation_to_idx[rev_rel] = len(self.relation_to_idx)
                self.idx_to_relation[self.relation_to_idx[rev_rel]] = rev_rel

                self.idx_to_rev_relation_idx[self.relation_to_idx[rel]] = self.relation_to_idx[rev_rel]
                self.idx_to_rev_relation_idx[self.relation_to_idx[rev_rel]] = self.relation_to_idx[rel]

            for subj, obj, _ in typed_relation_instances.relation_to_instances[rel]:
                if subj not in self.node_to_idx:
                    self.node_to_idx[subj] = len(self.node_to_idx)
                    self.idx_to_node[self.node_to_idx[subj]] = subj
                if obj not in self.node_to_idx:
                    self.node_to_idx[obj] = len(self.node_to_idx)
                    self.idx_to_node[self.node_to_idx[obj]] = obj