from main.data.TypedRelationInstances import TypedRelationInstances
from main.data.Vocabs import Vocabs
from main.data.Split import Split
from main.features.PRAPathReader import PRAPathReader
from main.features.PathReader import PathReader
from main.experiments.Metrics import score_cvsm

import os
import shutil
import subprocess
import time
import datetime
import json

# Improvement: add main to modify parameters in make_data_format.sh for only_relation and get_only_relation


class CVSMDriver:
    """
    This class helps to run the CVSM model implemented in torch using a dataset processed by this main base. The inputs
    to this class should be the data folder and the CVSM directory. Prior to running this code, PRA paths need to be
    created using PRA main and stored in /pra_paths folder.

    Right now, this driver supports two types of paths:

        1. relation paths extracted from random walk by PRA (PRA paths);
        2. paths with entities extracted from PathExtractor.pyx which uses BFS (BFS paths).

    Four kinds of CVSM models can be tested: REL, REL+ENT, REL+TYP, REL+ENT+TYP:

        1. REL model can be run with PRA paths by setting include_entity, has_entity, and include_entity_type to false. It can be run with BFS paths by setting include_entity and include_entity_type to false, and has_entity true
        2. REL+ENT model can be run with BFS paths by setting include_entity and has_entity to true, and include_entity_type false
        3. REL+TYP model can be run with BFS paths by setting include_entity_type and has_entity to true, and include_entity false
        4. REL+ENT+TYP model can be run with BFS paths by setting include_entity, has_entity, and include_entity_type true

    """
    def __init__(self, experiment_dir, cvsm_run_dir, dataset, include_entity=False, has_entity=False, augment_data=False,
                 include_entity_type=False):
        """
        Init

        :param experiment_dir: the folder for storing data in cvsm format
        :param cvsm_run_dir: where Das et al's csvm repo is located
        :param dataset: the name of the dataset, can be one of "wordnet", "robot", "freebase", or "fbclueweb"
        :param include_entity: whether to use entity information from input path data
        :param has_entity: whether input path data has entity information
        :param augment_data: whether input path data are augmented. Also see :meth:`main.features.PathExtractor` for
                             more details about data augmentation.
        :param include_entity_type: whether to use entity type information
        """
        # dirs
        self.experiment_dir = experiment_dir
        self.cvsm_run_dir = cvsm_run_dir
        self.cvsm_data_dir = None
        self.cvsm_result_dir = None

        # params
        self.include_entity = include_entity
        self.has_entity = has_entity
        self.include_entity_type = include_entity_type

        self.has_development_set = True

        self.augment_data = augment_data
        assert dataset == "wordnet" or dataset == "robot" or dataset == "freebase" or dataset == "fbclueweb"

        # ToDo: let the reader for the raw data create this param instead of specify it manually here
        if dataset == "wordnet":
            self.num_types = 14
        elif dataset == "robot":
            self.num_types = 10
        elif dataset == "freebase":
            self.num_types = 7
        elif dataset == "fbclueweb":
            self.num_types = 8

        self.relation_vocab_size = 0
        self.entity_vocab_size = 0

        if dataset == "fbclueweb":
            self.relation_vocab_size = 51390
            self.entity_vocab_size = 1542690

    def setup_cvsm_dir(self):
        """
        This function is used to set up cvsm directory in the data folder. This uses a path reader to help read paths
        and create paths and vocabs in the cvsm format.

        :return:
        """
        #################################################
        # -1. Set up files and directories
        domain_filename = os.path.join(self.experiment_dir, "domains.tsv")
        range_filename = os.path.join(self.experiment_dir, "ranges.tsv")
        edges_filename = os.path.join(self.experiment_dir, "edges.txt")
        split_dir = os.path.join(self.experiment_dir, "split")
        pra_path_dir = os.path.join(self.experiment_dir, "pra_paths")
        path_dir = os.path.join(self.experiment_dir, "paths")

        if self.augment_data:
            input("Warning: using augmented path. Press Enter to continue.")
            path_dir = os.path.join(self.experiment_dir, "paths_augment")

        if not self.include_entity:
            if not self.has_entity:
                cvsm_dir = os.path.join(self.experiment_dir, "cvsm")
            else:
                # Debug: make sure this uses the same data input as cvsm_entity
                cvsm_dir = os.path.join(self.experiment_dir, "cvsm_entity") # before is cvsm_bfs
        else:
            cvsm_dir = os.path.join(self.experiment_dir, "cvsm_entity")

        create_cvsm_folder = True
        if os.path.exists(cvsm_dir):
            answer = input("CVSM folder already exists. Recreate it? Y/N ")
            if answer == "Y":
                create_cvsm_folder = True
                shutil.rmtree(cvsm_dir)
                os.mkdir(cvsm_dir)
            elif answer == "N":
                create_cvsm_folder = False
            else:
                raise Exception("Please input Y or N")
        else:
            os.mkdir(cvsm_dir)

        if self.augment_data:
            self.cvsm_data_dir = os.path.join(cvsm_dir, "augment_data")
        else:
            self.cvsm_data_dir = os.path.join(cvsm_dir, "data")

        self.cvsm_result_dir = os.path.join(cvsm_dir, "results")

        ##################################################
        # 0. Process data

        # typed_relation_instances = TypedRelationInstances()
        # typed_relation_instances.read_domains_and_ranges(domain_filename, range_filename)
        # typed_relation_instances.construct_from_labeled_edges(edges_filename, entity_name_is_typed=False,
        #                                                       is_labeled=False)
        # vocabs = Vocabs()
        # vocabs.build_vocabs(typed_relation_instances)
        # split = Split()
        # split.read_splits(split_dir, vocabs, entity_name_is_typed=True)
        #
        # self.relation_vocab_size = len(vocabs.relation_to_idx)
        # self.entity_vocab_size = len(vocabs.node_to_idx)
        #
        # # check if has development set
        # for rel in split.relation_to_splits_to_instances:
        #     if "development" in split.relation_to_splits_to_instances[rel]:
        #         self.has_development_set = True
        #         break

        if create_cvsm_folder:
            typed_relation_instances = TypedRelationInstances()
            typed_relation_instances.read_domains_and_ranges(domain_filename, range_filename)
            typed_relation_instances.construct_from_labeled_edges(edges_filename, entity_name_is_typed=False,
                                                                  is_labeled=False)
            vocabs = Vocabs()
            vocabs.build_vocabs(typed_relation_instances)
            split = Split()
            split.read_splits(split_dir, vocabs, entity_name_is_typed=True)

            self.relation_vocab_size = len(vocabs.relation_to_idx)
            self.entity_vocab_size = len(vocabs.node_to_idx)

            # check if has development set
            for rel in split.relation_to_splits_to_instances:
                if "development" in split.relation_to_splits_to_instances[rel]:
                    self.has_development_set = True
                    break

            if not self.include_entity:
                if not self.has_entity:
                    print("Read PRA's pra features from pra_paths directory")
                    pra_path_reader = PRAPathReader(save_dir=pra_path_dir, include_entity=False)
                    pra_path_reader.read_paths(split)
                    pra_path_reader.write_cvsm_files(cvsm_dir=self.cvsm_data_dir, split=split, vocabs=vocabs)
                else:
                    print("Read paths with entities from paths directory")
                    path_reader = PathReader(save_dir=path_dir)
                    path_reader.read_paths(split)
                    entity2types_filename = os.path.join(self.experiment_dir, "entity2types.json")
                    path_reader.write_cvsm_files(self.cvsm_data_dir, split, vocabs, entity2types_filename)
            else:
                print("Read paths with entities from paths directory")
                path_reader = PathReader(save_dir=path_dir)
                path_reader.read_paths(split)
                entity2types_filename = os.path.join(self.experiment_dir, "entity2types.json")
                path_reader.write_cvsm_files(self.cvsm_data_dir, split, vocabs, entity2types_filename)

    def run(self):
        """
        This function is used to run cvsm original code. The cvsm original code will first vectorize paths and then
        train models on vectorized paths.

        :return:
        """

        self.setup_cvsm_dir()

        ###################################################
        # 1. CVSM main prepares data
        #    The main mainly group entity pairs with same number of paths together for batch training.
        print("CVSM run directory is", self.cvsm_run_dir)

        # a. copy vocab to cvsm run dir
        cvsm_side_vocab_dir = os.path.join(self.cvsm_run_dir, "vocab")

        # if os.path.exists(cvsm_side_vocab_dir):
        #     raise Exception("vocab dir in CVSM run dir already exists.")
        # this_side_vocab_dir = os.path.join(self.cvsm_data_dir, "vocab")
        # if not os.path.exists(this_side_vocab_dir):
        #     raise Exception("Can not find generated vocabs in", this_side_vocab_dir)
        # else:
        #     shutil.copytree(this_side_vocab_dir, cvsm_side_vocab_dir)

        # # b. copy input data to cvsm example dir
        # cvsm_side_data_input_dir = os.path.join(self.cvsm_run_dir, "data/examples/auto_generated_data_input")
        cvsm_side_data_output_dir = os.path.join(self.cvsm_run_dir, "data/examples/auto_generated_data_output")
        # if os.path.exists(cvsm_side_data_input_dir):
        #     shutil.rmtree(cvsm_side_data_input_dir)
        # if os.path.exists(cvsm_side_data_output_dir):
        #     shutil.rmtree(cvsm_side_data_output_dir)
        # this_side_data_input_dir = os.path.join(self.cvsm_data_dir, "data_input")
        # shutil.copytree(this_side_data_input_dir, cvsm_side_data_input_dir)

        # # c. configure data format
        # format_data_wkdir = os.path.join(self.cvsm_run_dir, "data")
        # config_filename = os.path.join(format_data_wkdir, "config.sh")
        # new_config_filename = os.path.join(format_data_wkdir, "auto_generated_config.sh")
        # with open(config_filename, "r") as fho:
        #     with open(new_config_filename, "w+") as fhn:
        #         for line in fho:
        #             new_line = ""
        #             if "mainDir" in line:
        #                 new_line = "mainDir='" + self.cvsm_run_dir + "'\n"
        #             elif "data_dir" in line:
        #                 new_line = "data_dir='examples/auto_generated_data_input'\n"
        #             elif "out_dir" in line:
        #                 new_line = "out_dir='examples/auto_generated_data_output'\n"
        #             elif "only_relation" in line[0:12]:
        #                 new_line = "only_relation=" + str(1-int(self.has_entity)) + "\n"
        #             elif "max_path_length" in line:
        #                 new_line = "max_path_length=8\n"
        #             elif "num_entity_types" in line:
        #                 new_line = "num_entity_types=" + str(self.num_types) + "\n"
        #             elif "get_only_relations" in line:
        #                 # Note: only when running on relations, the vectorized paths only include relations. All other
        #                 #       cases use vectorized paths including rel, ent, and typ.
        #                 if not self.include_entity and not self.include_entity_type:
        #                     new_line = "get_only_relations=1\n"
        #                 else:
        #                     new_line = "get_only_relations=0\n"
        #             else:
        #                 new_line = line
        #             fhn.write(new_line)
        #
        # # d. run `bash make_data_format.sh config.sh` to vectorize paths
        # os.chdir(format_data_wkdir)
        #
        # subprocess.check_call(['bash', 'make_data_format.sh', 'auto_generated_config.sh'])

        ####################################################
        # 2. Run CVSM model for each relation
        relation_to_scores = {}
        date_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        for cvsm_side_rel_dir in os.listdir(cvsm_side_data_output_dir):
            time.sleep(3)
            run_wkdir = os.path.join(self.cvsm_run_dir, "run_scripts")
            os.chdir(run_wkdir)
            rel = cvsm_side_rel_dir.split("/")[-1]
            print("\n\n" + "#" * 200)
            print("Run CVSM model for relation", rel)

            # a. modify train configs for training this relation
            cvsm_side_model_dir = os.path.join(run_wkdir,
                                               "results/" + date_time)
            config_filename = os.path.join(run_wkdir, "config.sh")
            new_config_filename = os.path.join(run_wkdir, "auto_generated_config.sh")
            with open(config_filename, "r") as fho:
                with open(new_config_filename, "w+") as fhn:
                    for line in fho:
                        new_line = ""
                        # Four cases:
                        # rel
                        if not self.include_entity and not self.include_entity_type:
                            if "data_dir" in line:
                                new_line = "data_dir='" + os.path.join(cvsm_side_data_output_dir,
                                                                       str(rel) + "/train.list") + "'\n"
                            elif "includeEntityTypes" in line:
                                new_line = "includeEntityTypes=" + str(0) + "\n"
                            elif "includeEntity" in line:
                                new_line = "includeEntity=" + str(0) + "\n"
                            elif "numFeatureTemplates" in line:
                                new_line = "numFeatureTemplates=" + str(1) + "\n"
                            elif "relationVocabSize" in line:
                                new_line = "relationVocabSize=" + str(self.relation_vocab_size + 1) + "\n"
                            elif "predicate_name" in line:
                                new_line = "predicate_name='" + str(rel) + "'\n"
                            elif "output_dir" in line:
                                new_line = "output_dir='" + cvsm_side_model_dir + "'\n"
                            else:
                                new_line = line
                        # rel + ent OR rel + type OR rel + ent + type
                        else:
                            if "data_dir" in line:
                                new_line = "data_dir='" + os.path.join(cvsm_side_data_output_dir,
                                                                       str(rel) + "/train.list") + "'\n"
                            elif "includeEntityTypes" in line:
                                new_line = "includeEntityTypes=" + str(int(self.include_entity_type)) + "\n"
                            elif "numEntityTypes" in line:
                                new_line = "numEntityTypes=" + str(self.num_types) + "\n"
                            elif "includeEntity" in line:
                                new_line = "includeEntity=" + str(int(self.include_entity)) + "\n"
                            elif "numFeatureTemplates" in line:
                                new_line = "numFeatureTemplates=" + str(2 + self.num_types) + "\n"
                            elif "relationVocabSize" in line:
                                # +1 for PAD_TOKEN, +1 for END_RELATION
                                new_line = "relationVocabSize=" + str(self.relation_vocab_size + 2) + "\n"
                            elif "entityTypeVocabSize" in line:
                                entity_type_vocab_filename = os.path.join(cvsm_side_vocab_dir, "entity_type_vocab.txt")
                                with open(entity_type_vocab_filename, "r") as ftype:
                                    entity_type_vocab = json.load(ftype)
                                new_line = "entityTypeVocabSize=" + str(entity_type_vocab["#PAD_TOKEN"] + 1) + "\n"
                            elif "entityVocabSize" in line:
                                new_line = "entityVocabSize=" + str(self.entity_vocab_size + 1) + "\n"
                            elif "predicate_name" in line:
                                new_line = "predicate_name='" + str(rel) + "'\n"
                            elif "output_dir" in line:
                                new_line = "output_dir='" + cvsm_side_model_dir + "'\n"
                            else:
                                new_line = line
                        fhn.write(new_line)

            # b. start training
            subprocess.check_call(['bash', 'train.sh', "auto_generated_config.sh"])

            # c. modify evaluation configs for evaluating this relation
            eval_wkdir = os.path.join(self.cvsm_run_dir, "eval")
            os.chdir(eval_wkdir)
            print("\n\n" + "#" * 200)
            print("Evaluate CVSM model for relation", rel)
            config_filename = os.path.join(eval_wkdir, "config.sh")
            new_config_filename = os.path.join(eval_wkdir, "auto_generated_config.sh")
            with open(config_filename, "r") as fho:
                with open(new_config_filename, "w+") as fhn:
                    for line in fho:
                        new_line = ""
                        if "data_dir" in line:
                            new_line = "data_dir='" + os.path.join(cvsm_side_data_output_dir, str(rel)) + "'\n"
                        # elif "relationVocabSize" in line:
                        #     new_line = "relationVocabSize=" + str(len(vocabs.relation_to_idx) + 1) + "\n"
                        elif "predicate_name" in line:
                            new_line = "predicate_name='" + str(rel) + "'\n"
                        elif "dir_path" in line:
                            new_line = "dir_path='" + cvsm_side_model_dir + "'\n"
                        else:
                            new_line = line
                        fhn.write(new_line)

            # hack to make cvsm work when we don't have data in dev split
            if not self.has_development_set:
                with open(os.path.join(cvsm_side_data_output_dir, os.path.join(cvsm_side_rel_dir, "dev.list")), "w+") as fh:
                    pass

            # d. start evaluating
            subprocess.check_call(['bash', 'get_accuracy_and_trec.sh', "auto_generated_config.sh"])

            # e. read evaluation results and generate scores
            best_val_acc = 0
            best = (0, 0, 0, 0)
            for epoch in range(51):
                result_filename = os.path.join(cvsm_side_model_dir, str(rel) + "/dev.scores.model-" + str(epoch))
                ap, rr, acc = score_cvsm(result_filename)
                if acc > best_val_acc:
                    best_val_acc = acc
                    result_filename = os.path.join(cvsm_side_model_dir, str(rel) + "/test.scores.model-" + str(epoch))
                    ap, rr, acc = score_cvsm(result_filename)
                    best = (epoch, ap, rr, acc)
            relation_to_scores[rel] = best

            # remove trained models for this relation to save space
            shutil.rmtree(os.path.join(cvsm_side_model_dir, str(rel)))
            os.system("rm -rf /home/weiyu/.local/share/Trash/*")

        print(relation_to_scores)
        aps = [relation_to_scores[rel][1] for rel in relation_to_scores]
        accs = [relation_to_scores[rel][3] for rel in relation_to_scores]
        print("MAP", sum(aps) / len(aps))
        print("Average Accuracy", sum(accs) / len(accs))