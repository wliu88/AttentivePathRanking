import os
import glob
from nltk.corpus import wordnet
import wikidata.client
import pickle
from tqdm import tqdm

########################################################################################################################
# DEPRECATED!

# We use all triples in train and test set to write one complete set of relation instances. The split of the
# original data will not be used. A new split will be constructed later.

# Important: freebase api is depleted, so I am using freebase2wikidata mapping to find entity name corresponding to each
#            machine id in the triple files. This mapping can be downloaded from https://developers.google.com/freebase/
#            Wikidata api also needs to installed via pip install Wikidata


class Freebase15kReader:
    def __init__(self, dir):
        self.dir = dir
        if not os.path.exists(dir):
            raise Exception(dir, "not exist")

        self.mid_to_wikid = {}
        self.wikid_to_name = {}
        # self.synsets = set()
        # self.synonyms = set()
        self.train_instances = []
        self.test_instances = []
        self.relations = set()

        self.client = wikidata.client.Client()

    def get_mid_to_name(self):
        mid2name = {}
        count=0
        definitions_filename = os.path.join(self.dir, "FB15k/fb2w.nt")
        # 1. definitions
        with open(definitions_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if "\t" not in line:
                    continue
                line = line[:-1].strip()
                mid, _, wikidata_url = line.split("\t")
                mid = "/" + mid[1:-1].split("/")[-1].replace(".", "/")
                wikid = wikidata_url[1:-1].split("/")[-1]

                try:
                    entity_name = str(self.client.get(wikid).label)
                    entity_name = "_".join(entity_name.lower().split(" "))
                    mid2name[mid] = entity_name
                    print("{}: {}".format(count, entity_name))
                    count += 1
                except:
                    print("Cannot retrive name")

        mid_to_name_filename = os.path.join(self.dir, "mid2name.pkl")
        pickle.dump(mid2name, open(mid_to_name_filename, "wb"))

    def read_data(self):
        """
        Two files in the FB15K folder are used.
        1. test.txt
        2. train.txt
        :param dir:
        :return:
        """
        files = glob.glob(os.path.join(self.dir, "FB15k/*.txt"))
        print(files)
        for file in files:
            if "train" in file:
                train_filename = file
            elif "test" in file:
                test_filename = file

        definitions_filename = os.path.join(self.dir, "FB15k/fb2w.nt")
        # 1. definitions
        with open(definitions_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if "\t" not in line:
                    continue
                line = line[:-1].strip()
                mid, _, wikidata_url = line.split("\t")
                mid = "/" + mid[1:-1].split("/")[-1].replace(".", "/")
                wikid = wikidata_url[1:-1].split("/")[-1]
                self.mid_to_wikid[mid] = wikid

        # 2. train file
        with open(train_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source, relation, target = line.split("\t")
                source = self.get_entity_name(source)
                target = self.get_entity_name(target)
                if source is not None and target is not None:
                    self.train_instances.append((source, relation, target))
                    self.relations.add(relation)
        print("There are", len(self.train_instances), "triplets in train data")

        # 3. test file
        with open(test_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source, relation, target = line.split("\t")
                source = self.get_entity_name(source)
                target = self.get_entity_name(target)
                if source is not None and target is not None:
                    self.test_instances.append((source, relation, target))
                    self.relations.add(relation)
        print("There are", len(self.test_instances), "triplets in test data")

        print("There are", len(self.relations), "unique relations in train and test")

        self.save_dictionaries()

    def no_matches(self):
        files = glob.glob(os.path.join(self.dir, "FB15k/*.txt"))
        print(files)
        for file in files:
            if "train" in file:
                train_filename = file
            elif "test" in file:
                test_filename = file
            elif "dev" in file:
                dev_filename = file

        unique_mids = set()
        with open(train_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source, relation, target = line.split("\t")
                unique_mids.update([source, target])

        # 3. test file
        with open(test_filename, "r") as fh:
            for line in fh:
                line = line.strip()
                if len(line) == 0:
                    continue
                source, relation, target = line.split("\t")
                unique_mids.update([source, target])

        count = 0
        for mid in unique_mids:
            name = self.get_entity_name(mid)
            if name is not None:
                count+=1
        print("unique mids", len(unique_mids))
        print("matching freebase name", count)

    def write_relation_domain_and_ranges(self):
        domains_filename = os.path.join(self.dir, "domains.tsv")
        ranges_filename = os.path.join(self.dir, "ranges.tsv")
        with open(domains_filename, "w+") as fd:
            with open(ranges_filename, "w+") as fr:
                for rel in self.relations:
                    fd.write(rel + "\tsynset\n")
                    fr.write(rel + "\tsynset\n")

    def write_edges(self):
        edges_filename = os.path.join(self.dir, "edges.txt")
        with open(edges_filename, "w+") as fe:
            for subj, rel, obj in self.train_instances:
                fe.write(subj + "\t" + rel + "\t" + obj + "\n")
            for subj, rel, obj in self.test_instances:
                fe.write(subj + "\t" + rel + "\t" + obj + "\n")

    def get_entity_name(self, mid):
        if mid not in self.mid_to_wikid:
            print(mid, "not in mid_to_wikid")
            return None
        wikid = self.mid_to_wikid[mid]
        if wikid in self.wikid_to_name:
            return self.wikid_to_name[wikid]
        else:
            try:
                entity_name = str(self.client.get(wikid).label)
                entity_name = "_".join(entity_name.lower().split(" "))
                self.wikid_to_name[wikid] = entity_name
                print(entity_name)
                return entity_name
            except:
                print("Cannot retrive name")
                return None

    def save_dictionaries(self):
        fb2name_filename = os.path.join(self.dir, "FB15k/mid2wiki_and_wiki2name.pkl")
        pickle.dump([self.mid_to_wikid, self.wikid_to_name], open(fb2name_filename, "wb"))

    def load_dictionaries(self):
        fb2name_filename = os.path.join(self.dir, "FB15k/mid2wiki_and_wiki2name.pkl")
        if not os.path.exists(fb2name_filename):
            return False
        self.mid_to_wikid, self.wikid_to_name = pickle.load(open(fb2name_filename, "rb"))
        return True


if __name__ == "__main__":
    fb = Freebase15kReader("/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/deprecated_datasets/freebase15k")
    fb.get_mid_to_name()
    # fb.read_data()
    # wn.write_relation_domain_and_ranges()
    # wn.write_edges()

    # fb.load_dictionaries()
    # fb.no_matches()
