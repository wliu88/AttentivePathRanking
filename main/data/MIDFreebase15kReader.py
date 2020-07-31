import os
import glob
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
import wikidata.client


# We use all triples in train and test set to write one complete set of relation instances. The split of the
# original data will not be used. A new split will be constructed later.


class MIDFreebase15kReader:
    """
    This class helps read raw FB15k-237 data and rewrite the data in the format necessary for the experiments.

    Data files involved:

    - input: ``/FB15k-237/train.txt``, ``/FB15k-237/test.txt``, ``/FB15k-237/valid.txt``, ``/FB15k-237/fb2w.nt``,
      ``/type_information/entity2type.txt``, ``/type_information/relation_specific.txt``
    - output: ``edges.txt``, ``domains.tsv``, ``ranges.tsv``, ``synonym2vec.pkl``, ``entity2types.json``,
      ``mid2name.pkl``

    .. note::

      We extract all triples in the original train and test set. The split of the original data is suitable for
      testing embedding methods and will not be used.

    .. note::
      The entities in the raw data are MIDs (Machine Identifiers), which are numbers.
    """

    def __init__(self, dir, filter=False, word2vec_filename=""):
        """
        Initialize a reader object.

        :param dir: the root of the data folder, where the ``/FB15k-237`` folder and the ``/type_information`` folder
            will be found, and all output files will be written to.
        :param filter: if set to True, entities with no corresponding word embeddings will be removed.
        :param word2vec_filename: the word embeddings for the entities.
        """
        self.dir = dir
        if not os.path.exists(dir):
            raise Exception(dir, "not exist")

        self.train_instances = []
        self.relations = set()
        self.mids = set()
        self.relation_domain = {}
        self.relation_range = {}

        self.filter = filter
        self.word2vec_filename = word2vec_filename

        self.synonym2vec = {}

        self.client = wikidata.client.Client()

    def read_data(self):
        """
        This function processes the raw data. Specifically, this function does the following few things:

        - It reads all relation instances in ``/FB15k-237`` and type information in ``/type_information``
          for entities and relations.
        - It removes relations and entities without type information.
        - It retrieves word embeddings for all entities and saves the embeddings in ``synonym2vec.pkl``.
        - It prints out relations with more than 3000 instances to help us select relations to test with.
        - It finds the 7 most frequently occurring types for each entity and save them in ``entity2types.json``.

        """

        # 1. collect all relevant files
        files = glob.glob(os.path.join(self.dir, "FB15k-237/*.txt"))
        for file in files:
            if "train" in file:
                train_filename = file
            elif "test" in file:
                test_filename = file
            elif "valid" in file:
                dev_filename = file

        files = glob.glob(os.path.join(self.dir, "type_information/*.txt"))
        for file in files:
            if "entity2type" in file:
                entity2type_filename = file
            elif "relation_specific" in file:
                relation_domain_range_filename = file

        # 2. read entity type and relation domain and range information
        mid2types = {}
        with open(entity2type_filename) as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                contents = line.split("\t")
                mid = contents[0]
                # important: bc "/" in relations maybe confused with "/" as dir
                mid = mid.replace("/", "|")
                types = contents[1:]
                mid2types[mid] = types

        relation_domain = {}
        relation_range = {}
        with open(relation_domain_range_filename) as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                relation, domain, range = line.split("\t")
                relation = relation.replace("/", "|")
                domain = domain.replace("/", "|")
                range = range.replace("/", "|")
                relation_domain[relation] = domain
                relation_range[relation] = range

        # 3. get all mids and relations in the data set
        mids = set()
        relations = set()
        for filename in [train_filename, test_filename, dev_filename]:
            with open(filename, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    source, relation, target = line.split("\t")
                    mids.add(source.replace("/", "|"))
                    mids.add(target.replace("/", "|"))
                    relations.add(relation.replace("/", "|"))

        # 4. remove mids with no type information, remove relations with no domain and range information
        # Note: No need to filter out entities without types. We also have done this for WN18RR dataset
        filtered_mids = set()
        for mid in mids:
            if mid in mid2types:
                filtered_mids.add(mid)
        print("mids before and after filtered by type information", len(mids), len(filtered_mids))

        filtered_relations = set()
        for relation in relations:
            if relation in relation_domain:
                filtered_relations.add(relation)
        print("relations before and after filtered by type information", len(relations), len(filtered_relations))

        # 5. Load word2vec and remove mids with no word2vec mapping. The mapping will be used for context-aware
        mids = filtered_mids
        filtered_mids = set()
        if self.filter:
            if os.path.exists(os.path.join(self.dir, "synonym2vec.pkl")):
                print("synonym2vec pickle file found: ", os.path.join(self.dir, "synonym2vec.pkl"))
                self.synonym2vec = pickle.load(open(os.path.join(self.dir, "synonym2vec.pkl"), "rb"))
            else:
                if os.path.exists(self.word2vec_filename):
                    # Important: only native word2vec file needs binary flag to be true
                    word2vec_model = KeyedVectors.load_word2vec_format(self.word2vec_filename, binary=True)
                else:
                    raise Exception("word2vec file not found")
                print("word2vec loaded")

                for mid in mids:
                    try:
                        self.synonym2vec[mid] = word2vec_model.get_vector(mid.replace("|", "/"))
                        filtered_mids.add(mid)
                    except KeyError:
                        pass
                print(len(mids) - len(filtered_mids), "MIDs have no matching word2vec entry")
                print(len(filtered_mids), "MIDs have.")

                print("Saving synonym2vec to pickle file:", os.path.join(self.dir, "synonym2vec.pkl"))
                pickle.dump(self.synonym2vec, open(os.path.join(self.dir, "synonym2vec.pkl"), "wb"))

        self.relations = filtered_relations
        self.mids = filtered_mids
        self.relation_domain = relation_domain
        self.relation_range = relation_range

        # 6. Read data.
        # keep track of instances of relations
        relation_to_instance_count = {}
        for filename in [train_filename, test_filename, dev_filename]:
            with open(filename, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    source, relation, target = line.split("\t")
                    source = source.replace("/", "|")
                    relation = relation.replace("/", "|")
                    target = target.replace("/", "|")
                    if source in filtered_mids and target in filtered_mids and relation in filtered_relations:
                        self.train_instances.append((source, relation, target))
                        if relation not in relation_to_instance_count:
                            relation_to_instance_count[relation] = 0
                        relation_to_instance_count[relation] += 1
        print("There are", len(self.train_instances), "triplets in data")
        print("There are", len(relation_to_instance_count), "relations")
        print("There are", len(self.mids), "entities")

        print("Relations with more than 3000 instances")
        for relation in relation_to_instance_count:
            if relation_to_instance_count[relation] > 1000:
                print("\"" + relation + "\",", end=' ')

        # 7. Write entity2types.json
        # (1). find occurrences of mid
        type2count = {}
        for mid in mid2types:
            types = mid2types[mid]
            for type in types:
                if type not in type2count:
                    type2count[type] = 0
                type2count[type] += 1

        # (2). only use 7 most occurring types
        types_used = set()
        entity2types = {}
        for mid in filtered_mids:
            all_types = mid2types[mid]
            all_types_with_count = []
            for type in all_types:
                all_types_with_count.append((type, type2count[type]))
            all_types_with_count.sort(key=lambda x: x[1], reverse=True)
            most_occurring_types = [type[0] for type in all_types_with_count][:7]
            # we arrange the types from least occurring to most occurring
            most_occurring_types.reverse()
            # print(most_occurring_types)
            entity2types[mid] = most_occurring_types
            types_used.update(most_occurring_types)

        print("Display entity types statistics, after filtering")
        total_length = []
        for node in entity2types:
            total_length.append(len(entity2types[node]))
        plt.hist(np.array(total_length), bins=14)
        plt.show()
        print("average number of types is", sum(total_length) / len(total_length))
        print("max number of types for an entity is", max(total_length))
        print("Number of types:", len(types_used))

        print("Writing entity2types to file")
        with open(os.path.join(self.dir, "entity2types.json"), "w+") as fh:
            json.dump(entity2types, fh)

    def get_mid_to_name(self):
        """
        This function retrieves names of entities from their definitions in ``/FB15k-237/fb2w.nt`` and
        saves to ``mid2name.pkl``.
        """

        # 1. collect all relevant files
        files = glob.glob(os.path.join(self.dir, "FB15k-237/*.txt"))
        for file in files:
            if "train" in file:
                train_filename = file
            elif "test" in file:
                test_filename = file
            elif "valid" in file:
                dev_filename = file

        mids = set()
        for filename in [train_filename, test_filename, dev_filename]:
            with open(filename, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    source, relation, target = line.split("\t")
                    mids.add(source.replace("/", "|"))
                    mids.add(target.replace("/", "|"))

        mid2name = {}
        count = 0
        definitions_filename = os.path.join(self.dir, "FB15k-237/fb2w.nt")
        # 1. definitions
        with open(definitions_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if "\t" not in line:
                    continue
                line = line[:-1].strip()
                mid, _, wikidata_url = line.split("\t")
                mid = "/" + mid[1:-1].split("/")[-1].replace(".", "/")
                mid = mid.replace("/", "|")
                # only get names of mids that are in the data
                if mid not in mids:
                    continue
                wikid = wikidata_url[1:-1].split("/")[-1]

                try:
                    entity_name = str(self.client.get(wikid).label)
                    entity_name = "_".join(entity_name.lower().split(" "))
                    mid2name[mid] = entity_name
                    print("{}/{}: {} <---> {}".format(count, len(mids), mid, entity_name))
                    count += 1
                except:
                    print("Cannot retrive name for {}".format(mid))

        mid_to_name_filename = os.path.join(self.dir, "mid2name.pkl")
        pickle.dump(mid2name, open(mid_to_name_filename, "wb"))

    def write_relation_domain_and_ranges(self):
        """
        This function writes relations' domains and ranges in ``domains.tsv`` and ``ranges.tsv``.
        """
        domains_filename = os.path.join(self.dir, "domains.tsv")
        ranges_filename = os.path.join(self.dir, "ranges.tsv")
        with open(domains_filename, "w+") as fd:
            with open(ranges_filename, "w+") as fr:
                for rel in self.relations:
                    fd.write(rel + "\t" + self.relation_domain[rel] + "\n")
                    fr.write(rel + "\t" + self.relation_range[rel] + "\n")

    def write_edges(self):
        """
        This function writes all processed relation instances in ``edges.txt``.
        """
        edges_filename = os.path.join(self.dir, "edges.txt")
        with open(edges_filename, "w+") as fe:
            for subj, rel, obj in self.train_instances:
                fe.write(subj + "\t" + rel + "\t" + obj + "\n")
