import os
import glob
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json

# https://stackoverflow.com/questions/42038337/what-is-the-connection-or-difference-between-lemma-and-synset-in-wordnet
# Important: Base on the information above, should we keep each lemma as a unique entity or each synonym?
# we are treating each synonyms as a unique entity now. So synonyms in a synset are different entities.
# The answer is it doesn't matter, because each synonym in this dataset is a unique synset. The number
# of unique synonym is equal to the number of unique synset

# We use all triples in train and test set to write one complete set of relation instances. The split of the
# original data will not be used. A new split will be constructed later.


class WordnetReader:
    """
    This class helps read raw WN18RR data and rewrite the data in the format necessary for the experiments.

    Data files involved:

    - input: ``/WN18RR/train.txt``, ``/WN18RR/test.txt``, ``/WN18RR/valid.txt``,
      ``/WN18RR/wordnet-mlj12-definitions.txt``
    - output: ``edges.txt``, ``domains.tsv``, ``ranges.tsv``, ``synonym2vec.pkl``, ``entity2types.json``,
      ``entity_type2vec.pkl``

    .. note::

      We extract all triples in the original train and test set. The split of the original data is suitable for
      testing embedding methods and will not be used.
    """

    def __init__(self, dir, filter=False, word2vec_filename="", remove_repetitions=False):
        """
        Initialize a reader object.

        :param dir: the root of the data folder, where the ``/WN18RR`` folder will be found, and all output files will
            be written to.
        :param filter: If True, we ignore entities that have no matching embedding in word2vec.
        :param word2vec_filename: the word embeddings for the entities.
        :param remove_repetitions: If True, remove repetition of relation instances such as a->rel1->b and b->rel1->a.
        """

        self.dir = dir
        if not os.path.exists(dir):
            raise Exception(dir, "not exist")

        self.idx_to_synonym = {}
        # self.synsets = set()
        # self.synonyms = set()
        self.train_instances = set()
        self.relations = set()

        self.filter = filter
        self.word2vec_filename = word2vec_filename
        self.word2vec_model = None

        self.synonym2vec = {}

        self.entity2types = {}
        self.entity_type2vec = {}

        # Important: This is used to remove repetition of relation instances such as a->rel1->b and b->rel1->a
        self.remove_repetitions = remove_repetitions

    def read_data(self):
        """
        This function processes the raw data.
        """
        # files = glob.glob(os.path.join(self.dir, "wordnet-mlj12/*.txt"))
        files = glob.glob(os.path.join(self.dir, "WN18RR/*.txt"))
        for file in files:
            if "definitions" in file:
                definitions_filename = file
            elif "train" in file:
                train_filename = file
            elif "test" in file:
                test_filename = file
            elif "valid" in file:
                dev_filename = file

        # 1. definitions
        with open(definitions_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                idx, synonym, _ = line.split("\t")
                idx = int(idx)

                # change the format of synonym from "__name_pos_nn" to "name.pos.nn", which is used in nltk.wordnet
                # pos: NN -> n, VB -> v, JJ -> s,
                # nn: 1 -> 01, 10 -> 10

                # NLTK documentation:
                # { Part-of-speech constants
                # ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
                # }
                # ADJ_SAT is adjective satellite
                content = synonym.replace("__", "").split("_")
                new_name = "_".join(content[0:-2]).lower()
                synonym_pos = content[-2]
                if synonym_pos == "NN":
                    new_pos = "n"
                elif synonym_pos == "VB":
                    new_pos = "v"
                elif synonym_pos == "JJ":
                    new_pos = "a"
                elif synonym_pos == "RB":
                    new_pos = "r"
                else:
                    print(content)
                    raise Exception("Synonym position\"", synonym_pos, "\"is not recognized.")
                synonym_sense = int(content[-1])
                # add leading 0
                new_sense = "{:02d}".format(synonym_sense)
                synonym = ".".join([new_name, new_pos, new_sense])

                self.idx_to_synonym[idx] = synonym
        #         self.synsets.add(wordnet.synset(synonym))
        #         self.synonyms.add(synonym)
        # print(len(self.synsets))
        # print(len(self.synonyms))
        print("There are", len(self.idx_to_synonym), "unique synset in definitions.")

        # 2. remove entities that have no mapping in word2vec
        if self.filter:
            if os.path.exists(os.path.join(self.dir, "synonym2vec.pkl")):
                print("synonym2vec pickle file found: ", os.path.join(self.dir, "synonym2vec.pkl"))
                self.synonym2vec = pickle.load(open(os.path.join(self.dir, "synonym2vec.pkl"), "rb"))
            else:
                if os.path.exists(self.word2vec_filename):
                    # Important: only native word2vec file needs binary flag to be true
                    if self.word2vec_model is None:
                        self.word2vec_model = KeyedVectors.load_word2vec_format(self.word2vec_filename, binary=True)
                else:
                    raise Exception("word2vec file not found")
                # print("Caching entity2vec for", len(self.idx_to_synonym), "synonym")
                no_match_count = 0
                for idx in self.idx_to_synonym:
                    synonym = self.idx_to_synonym[idx]
                    # Important: some wordnet words have -, which will be confused with delimiter for relations in a path.
                    #            e.g., up-to-dateness.n.01
                    if "-" in synonym:
                        no_match_count += 1
                        continue
                    # The following code tries to the word embedding corresponding to a synonym.
                    word = synonym[:-5]
                    try_another = True
                    # try the word
                    if try_another:
                        try:
                            self.synonym2vec[synonym] = self.word2vec_model.get_vector(word)
                            try_another = False
                        except KeyError:
                            try_another = True
                    # try capitalize. capitalize all words in compound words
                    if try_another:
                        try:
                            replace_word = "_".join([w.capitalize() for w in word.split("_")])
                            self.synonym2vec[synonym] = self.word2vec_model.get_vector(replace_word)
                            try_another = False
                        except KeyError:
                            try_another = True
                    # try all synonyms of the word
                    if try_another:
                        for syn in wordnet.synset(synonym).lemma_names():
                            if try_another:
                                try:
                                    self.synonym2vec[synonym] = self.word2vec_model.get_vector(syn)
                                    try_another = False
                                except KeyError:
                                    try_another = True
                    # no match
                    if try_another:
                        # print(entity_name)
                        no_match_count += 1

                print(no_match_count, "synonyms have no matching word2vec entry")
                print("Saving synonym2vec to pickle file:", os.path.join(self.dir, "synonym2vec.pkl"))
                pickle.dump(self.synonym2vec, open(os.path.join(self.dir, "synonym2vec.pkl"), "wb"))

        print("There are {} unique entities after removing entities without word2vec embeddings".format(len(self.synonym2vec)))

        # 2. train file
        with open(train_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source_idx, relation, target_idx = line.split("\t")
                relation = relation[1:]
                source_idx = int(source_idx)
                target_idx = int(target_idx)
                if self.idx_to_synonym[source_idx] in self.synonym2vec and self.idx_to_synonym[target_idx] in self.synonym2vec:
                    if self.remove_repetitions:
                        if (self.idx_to_synonym[target_idx], relation, self.idx_to_synonym[source_idx]) in self.train_instances:
                            continue
                    self.train_instances.add((self.idx_to_synonym[source_idx], relation, self.idx_to_synonym[target_idx]))
                    self.relations.add(relation)

        # 3. test file
        with open(test_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source_idx, relation, target_idx = line.split("\t")
                relation = relation[1:]
                source_idx = int(source_idx)
                target_idx = int(target_idx)
                if self.idx_to_synonym[source_idx] in self.synonym2vec and self.idx_to_synonym[target_idx] in self.synonym2vec:
                    if self.remove_repetitions:
                        if (self.idx_to_synonym[target_idx], relation, self.idx_to_synonym[source_idx]) in self.train_instances:
                            continue
                    self.train_instances.add((self.idx_to_synonym[source_idx], relation, self.idx_to_synonym[target_idx]))
                    self.relations.add(relation)

        # 4. dev file
        with open(dev_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source_idx, relation, target_idx = line.split("\t")
                relation = relation[1:]
                source_idx = int(source_idx)
                target_idx = int(target_idx)
                if self.idx_to_synonym[source_idx] in self.synonym2vec and self.idx_to_synonym[target_idx] in self.synonym2vec:
                    if self.remove_repetitions:
                        if (self.idx_to_synonym[target_idx], relation, self.idx_to_synonym[source_idx]) in self.train_instances:
                            continue
                    self.train_instances.add((self.idx_to_synonym[source_idx], relation, self.idx_to_synonym[target_idx]))
                    self.relations.add(relation)

        print("There are", len(self.relations), "unique relations in train, test, and dev")
        print("There are", len(self.train_instances), "relation instances")

    def get_entity_types(self):
        """
        This method will use wordnet to find types for each entity. The entity2type dictionary will be saved in
        ``/entity2types.json``. This method will also use word2vec model to eliminate types that have no mappings.
        Found mappings stored in entity_type2vec dictionary will be saved in ``/entity_type2vec.pkl``
        """
        count = 0
        for entity in self.synonym2vec:
            # all entities in this dataset are wordnet synsets
            synset = wordnet.synset(entity)
            all_hypernyms = []
            while True:
                hypernyms = synset.hypernyms()
                if not hypernyms:
                    break
                synset = hypernyms[0]
                all_hypernyms.append(synset.name())
            if not all_hypernyms:
                count += 1
            self.entity2types[entity] = all_hypernyms
        print(count, "entities out of", len(self.synonym2vec), "have no type at all")

        print("Display entity types statistics, before filtering")
        total_length = []
        for entity in self.entity2types:
            total_length.append(len(self.entity2types[entity]))
        plt.hist(np.array(total_length), bins=18)
        plt.show()

        # IMPORTANT: filter out entity types that have no matching word2vec entry
        print("Caching entity2vec for entity types")
        if self.word2vec_model is None:
            self.word2vec_model = KeyedVectors.load_word2vec_format(self.word2vec_filename, binary=True)
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
                    # print("There is no match for type:", type)
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
        print("average number of types:", sum(total_length) / len(total_length))
        plt.hist(np.array(total_length), bins=14)
        plt.show()
        print("max number of types for an entity is", max(total_length))

        print("Saving entity_type2vec to pickle file:", os.path.join(self.dir, "entity_type2vec.pkl"))
        pickle.dump(self.entity_type2vec, open(os.path.join(self.dir, "entity_type2vec.pkl"), "wb"))

        print("Writing entity2types to file")
        with open(os.path.join(self.dir, "entity2types.json"), "w+") as fh:
            json.dump(self.entity2types, fh)

    def write_relation_domain_and_ranges(self):
        """
        This function writes relations' domains and ranges in ``domains.tsv`` and ``ranges.tsv``.
        """
        domains_filename = os.path.join(self.dir, "domains.tsv")
        ranges_filename = os.path.join(self.dir, "ranges.tsv")
        with open(domains_filename, "w+") as fd:
            with open(ranges_filename, "w+") as fr:
                for rel in self.relations:
                    fd.write(rel + "\tsynset\n")
                    fr.write(rel + "\tsynset\n")

    def write_edges(self):
        """
        This function writes all processed relation instances in ``edges.txt``.
        """
        edges_filename = os.path.join(self.dir, "edges.txt")
        with open(edges_filename, "w+") as fe:
            for subj, rel, obj in self.train_instances:
                fe.write(subj + "\t" + rel + "\t" + obj + "\n")


if __name__ == "__main__":
    wn = WordnetReader("/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/wordnet2",
                       filter=True,
                       word2vec_filename="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/word2vec/GoogleNews-vectors-negative300.bin")
    wn.read_data()
    wn.get_entity_types()
    # wn.write_relation_domain_and_ranges()
    # wn.write_edges()