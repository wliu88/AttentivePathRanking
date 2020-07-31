import collections
import os
import shutil


class TypedRelationInstances:
    """
    This class manages typed relation instances and meta-data for relations. *Typed* means that each relation has a
    domain and a range in terms of entity types.

    :ivar relation_domain: a dict mapping from a relation to its domain as an entity type
    :ivar relation_range: a dict mapping from a relation to its range as an entity type
    :ivar type_to_entities: a dict mapping from an entity type to a set of entities with that type
    :ivar relation_to_instances: an ordered dict mapping from a relation to all relation instances with that relation.
        Each instance is a tuple of (subject, object, label). Label is 1 or -1.
    """

    def __init__(self):
        self.relation_domain = {}
        self.relation_range = {}
        self.type_to_entities = {}
        # each instance is a tuple of (subject, object, label). label is 1 or -1.
        self.relation_to_instances = collections.OrderedDict()

    def read_domains_and_ranges(self, domain_filename, range_filename):
        """
        This function reads a domain file and a range file to initialize domains and ranges for all relations, and
        entity types.

        :param domain_filename: the name of the file that stores relation domain information
        :param range_filename: the name of the file that stores relation range information
        """
        with open(domain_filename, "r") as fh:
            for line in fh:
                line = line.replace("\n", "")
                if len(line) == 0:
                    continue
                # Debug: not able to recognize tab between
                rel, entity_type = line.split()

                if rel in self.relation_domain:
                    raise Exception(rel, 'already has', self.relation_domain[rel], 'as domain.')
                self.relation_domain[rel] = entity_type
                self.type_to_entities[entity_type] = set()
                self.relation_to_instances[rel] = []
        with open(range_filename, "r") as fh:
            for line in fh:
                line = line.replace("\n", "")
                if len(line) == 0:
                    continue
                rel, entity_type = line.split("\t")

                if rel in self.relation_range:
                    raise Exception(rel, 'already has', self.relation_domain[rel], 'as range.')
                self.relation_range[rel] = entity_type
                self.type_to_entities[entity_type] = set()
                self.relation_to_instances[rel] = []

    def construct_from_labeled_edges(self, filename, entity_name_is_typed, is_labeled):
        """
        This function initializes relation instances from a file.

        :param filename: the name of the file that stores all relation instances.
        :param entity_name_is_typed: True if type is included in each entity name. e.g., object:sandwich.n.01
        :param is_labeled: True if each relation instance has a label. Otherwise, all relation instances from file is
               true.
        """
        if is_labeled:
            print("Negative relation instances will be ignored. We will use PRA to generate negative relation instances.")

        with open(filename, "r") as fh:
            for line in fh:
                line = line.replace("\n", "")
                if len(line) == 0:
                    continue
                if is_labeled:
                    subj, rel, obj, label = line.split("\t")
                    label = int(label)
                    assert(label == -1 or label == 1)
                else:
                    label = 1
                    subj, rel, obj = line.split("\t")

                if (rel not in self.relation_domain) or (rel not in self.relation_range):
                    raise Exception("Domain or range for", rel, "has not been defined.")
                domain = self.relation_domain[rel]
                range = self.relation_range[rel]
                if not entity_name_is_typed:
                    subj = domain + ":" + subj
                    obj = range + ":" + obj
                self.type_to_entities[domain].add(subj)
                self.type_to_entities[range].add(obj)
                self.relation_to_instances[rel].append(tuple([subj, obj, label]))

    def write_to_pra_format(self, save_dir, only_positive_instance=True):
        """
        This function writes files containing relation instances and meta-data in the PRA format to a folder.

        :param save_dir: the output folder.
        :param only_positive_instance: if true, only writes true relation instances.

        .. note:: See code comments for more details about files in PRA format.
        """
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

        # 1. write relation instances as labeled edges.
        # The format is [entity1]\t[relation]\t[entity2]\n
        # each entity should be prefixed with it's type. e.g., object:bowl.n.01
        labeled_edges = os.path.join(save_dir, "labeled_edges.tsv")
        with open(labeled_edges, "w+") as fh:
            for rel in self.relation_to_instances:
                for subj, obj, label in self.relation_to_instances[rel]:
                    if only_positive_instance:
                        if label != 1:
                            continue
                    fh.write(subj + "\t" + rel + "\t" + obj + "\n")

        # 2. write relation instances for each relation.
        # The format is [entity1]\t[entity2]\n
        relations_dir = os.path.join(save_dir, "relations")
        os.mkdir(relations_dir)
        for rel in self.relation_to_instances:
            relation_file = os.path.join(relations_dir, rel)
            with open(relation_file, "w+") as fh:
                for subj, obj, label in self.relation_to_instances[rel]:
                    if only_positive_instance:
                        if label != 1:
                            continue
                    fh.write(subj + "\t" + obj + "\n")

        # 3. write ranges and domains.
        # Format is [relation]\t[domain],
        # where domain must have a corresponding file in category_instances/, which is a directory containing all of
        # the instances of each category, one category per file (format is just one instance per line).
        domains = os.path.join(save_dir, "domains.tsv")
        ranges = os.path.join(save_dir, "ranges.tsv")
        with open(domains, "w+") as fd:
            with open(ranges, "w+") as fr:
                for rel in self.relation_to_instances:
                    domain_name = self.relation_domain[rel]
                    range_name = self.relation_range[rel]
                    fd.write(rel + "\t" + domain_name + "\n")
                    fr.write(rel + "\t" + range_name + "\n")

        # write category instances
        category_dir = os.path.join(save_dir, "category_instances")
        os.mkdir(category_dir)
        for type in self.type_to_entities:
            type_instances = os.path.join(category_dir, type)
            with open(type_instances, "w+") as fti:
                for entity in self.type_to_entities[type]:
                    fti.write(entity + "\n")

