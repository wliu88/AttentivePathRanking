import os
import glob
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json


class RobotDataReader:
    """
    This class helps to read robot data and generate filtered edges, relation domain, relation range, synonym2vec,
    entity2types, and type2vec.

    Filter: we remove entities that have no word2vec or no hypernynms in wordnet.
            we remove out entity types that have no word2vec
    """
    def __init__(self, dir, word2vec_filename=""):
        self.dir = dir
        if not os.path.exists(dir):
            raise Exception(dir, "not exist")

        self.train_instances = []
        self.relations = set()
        self.entities = set()
        # a dictionary from entity to its' type hierarchy, which is a list of it's inherited hypernyms in wordnet. The
        # list goes from most specific to most abstract type.
        self.entity2types = {}

        # word vectors
        self.word2vec_filename = word2vec_filename
        print("Loading word2vec model")
        if os.path.exists(self.word2vec_filename):
            # Important: only native word2vec file needs binary flag to be true
            self.word2vec_model = KeyedVectors.load_word2vec_format(self.word2vec_filename, binary=True)
        else:
            raise Exception("word2vec file not found")

        # storing entity name to word2vec vector
        self.entity2vec = {}
        # storing entity type name to word2vec vector
        self.entity_type2vec = {}

    def read_data(self):
        """
        Read raw data from edges.txt.
        :param dir:
        :return:
        """
        # files = glob.glob(os.path.join(self.dir, "wordnet-mlj12/*.txt"))
        files = glob.glob(os.path.join(self.dir, "raw/*.txt"))
        for file in files:
            if "edges" in file:
                edges_filename = file

        # 1. edges
        with open(edges_filename, "r") as fh:
            for line in fh:
                line = line.replace("\n", "")
                if len(line) == 0:
                    continue
                subj, rel, obj, label = line.split("\t")
                label = int(label)
                assert (label == -1 or label == 1 or label == 0)

                # IMPORTANT: 3 FILTERING RULES
                if label == 0 or label == -1:
                    continue
                # IMPORTANT: we are ignoring whisking.n.00, chopping.n.00, vacuuming.n.00 bc they have no matching
                #            wordnet synsets.
                if "00" in subj or "00" in obj:
                    continue
                # IMPORTANT: ignoring "near" relation because of large out-degree
                if rel == "near":
                    continue

                self.train_instances.append((subj, rel, obj))
                self.entities.update([subj, obj])
                self.relations.add(rel)

        # 2. cache entity2vec
        print("Caching entity2vec for", len(self.entities), "entities")
        no_match_count = 0
        for entity in self.entities:
            # each entity in this dataset is a wordnet synset
            # remove pos and sense number
            word = entity[:-5]
            try_another = True
            # try the word
            if try_another:
                try:
                    self.entity2vec[entity] = self.word2vec_model.get_vector(word)
                    try_another = False
                except KeyError:
                    try_another = True
            # try capitalize. capitalize all words in compound words
            if try_another:
                try:
                    replace_word = "_".join([w.capitalize() for w in word.split("_")])
                    self.entity2vec[entity] = self.word2vec_model.get_vector(replace_word)
                    try_another = False
                except KeyError:
                    try_another = True
            # try all synonyms of the word
            if try_another:
                for syn in wordnet.synset(entity).lemma_names():
                    if try_another:
                        try:
                            self.entity2vec[entity] = self.word2vec_model.get_vector(syn)
                            try_another = False
                        except KeyError:
                            try_another = True
            if try_another:
                print("Use hardcode match of robot data for", entity)
                try:
                    if word == "water_bottle":
                        replace_word = "bottle"
                    elif word == "cutting_board":
                        replace_word = "chopping"
                    self.entity2vec[entity] = self.word2vec_model.get_vector(replace_word)
                    try_another = False
                except KeyError:
                    try_another = True
            # no match
            if try_another:
                print("There is no match for", entity)
                no_match_count += 1
        print(no_match_count, "entities have no matching word2vec entry")
        # There should be no entities that have no word2vec mapping
        assert no_match_count == 0

        print("Saving entity2vec to pickle file:", os.path.join(self.dir, "entity2vec.pkl"))
        pickle.dump(self.entity2vec, open(os.path.join(self.dir, "entity2vec.pkl"), "wb"))

    def get_entity_types(self):
        """
        This method will use wordnet to find types for each entity. The entity2type dictionary will be saved in a json
        file. This method will also use word2vec model to eliminate types that have no mappings. Found mappings stored
        in entity_type2vec dictionary will be saved in a pickle file.
        :return:
        """
        for entity in self.entities:
            # all entities in robot dataset are wordnet synsets
            synset = wordnet.synset(entity)
            all_hypernyms = []
            while True:
                hypernyms = synset.hypernyms()
                if not hypernyms:
                    break
                synset = hypernyms[0]
                all_hypernyms.append(synset.name())
            self.entity2types[entity] = all_hypernyms

        print("Display entity types statistics, before filtering")
        total_length = []
        for entity in self.entity2types:
            total_length.append(len(self.entity2types[entity]))
        plt.hist(np.array(total_length), bins=5)
        plt.show()

        # IMPORTANT: filter out entity types that have no matching word2vec entry
        print("Caching entity2vec for entity types")
        no_match_types = set()
        for entity in self.entity2types:
            filtered_types = []
            for type in self.entity2types[entity]:
                # remove pos and sense number
                word = type[:-5]
                try_another = True
                # try the word
                if try_another:
                    try:
                        self.entity_type2vec[type] = self.word2vec_model.get_vector(word)
                        filtered_types.append(type)
                        try_another = False
                    except KeyError:
                        try_another = True
                # try capitalize. capitalize all words in compound words
                if try_another:
                    try:
                        replace_word = "_".join([w.capitalize() for w in word.split("_")])
                        self.entity_type2vec[type] = self.word2vec_model.get_vector(replace_word)
                        filtered_types.append(type)
                        try_another = False
                    except KeyError:
                        try_another = True
                # try all synonyms of the word
                if try_another:
                    for syn in wordnet.synset(type).lemma_names():
                        if try_another:
                            try:
                                self.entity_type2vec[type] = self.word2vec_model.get_vector(syn)
                                filtered_types.append(type)
                                try_another = False
                            except KeyError:
                                try_another = True
                # no match
                if try_another:
                    print("There is no match for type:", type)
                    no_match_types.add(type)
            self.entity2types[entity] = filtered_types
        print(len(no_match_types), "types have no matching word2vec entry")

        # sanity check
        entity_type_vocab = {}
        for entity in self.entity2types:
            types = self.entity2types[entity]
            # a. construct type vocab
            for type in types:

                assert type not in no_match_types

                if type not in entity_type_vocab:
                    entity_type_vocab[type] = len(entity_type_vocab)
        print(len(entity_type_vocab), len(self.entity_type2vec))
        assert len(entity_type_vocab) == len(self.entity_type2vec)

        print("Display entity types statistics, after filtering")
        total_length = []
        for node in self.entity2types:
            total_length.append(len(self.entity2types[node]))
        plt.hist(np.array(total_length), bins=5)
        plt.show()

        print("Saving entity_type2vec to pickle file:", os.path.join(self.dir, "entity_type2vec.pkl"))
        pickle.dump(self.entity_type2vec, open(os.path.join(self.dir, "entity_type2vec.pkl"), "wb"))

        print("Writing entity2types to file")
        with open(os.path.join(self.dir, "entity2types.json"), "w+") as fh:
            json.dump(self.entity2types, fh)

    def write_relation_domain_and_ranges(self):
        domains_filename = os.path.join(self.dir, "domains.tsv")
        ranges_filename = os.path.join(self.dir, "ranges.tsv")
        with open(domains_filename, "w+") as fd:
            with open(ranges_filename, "w+") as fr:
                for rel in self.relations:
                    fd.write(rel + "\tobject\n")
                    if rel == "in" or rel == "on":
                        fr.write(rel + "\tlocation\n")
                    elif rel == "made_of":
                        fr.write(rel + "\tmaterial\n")
                    elif rel == "used_for":
                        fr.write(rel + "\taffordance\n")
                    else:
                        raise Exception("relation type", rel, "is not recognized.")

    def write_edges(self):
        edges_filename = os.path.join(self.dir, "edges.txt")
        with open(edges_filename, "w+") as fe:
            for subj, rel, obj in self.train_instances:
                fe.write(subj + "\t" + rel + "\t" + obj + "\n")


if __name__ == "__main__":
    rr = RobotDataReader("/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/robot",
                         word2vec_filename="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/word2vec/GoogleNews-vectors-negative300.bin")
    rr.read_data()
    rr.get_entity_types()
    # rr.write_relation_domain_and_ranges()
    # rr.write_edges()


