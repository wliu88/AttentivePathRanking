# import and build cython
import pyximport
pyximport.install()

from main.data.MIDFreebase15kReader import MIDFreebase15kReader
from main.data.TypedRelationInstances import TypedRelationInstances
from main.data.Vocabs import Vocabs
from main.graphs.AdjacencyGraph import AdjacencyGraph
from main.features.PathExtractor import PathExtractor
from main.features.CPRPathExtractorMP import CPRPathExtractorMP
from main.data.Split import Split
from main.features.PathReader import PathReader
from main.algorithms.PathRankingAlgorithm import PathRankingAlgorithm
from main.features.PRAPathReader import PRAPathReader
from main.experiments.Metrics import score_cvsm
from main.experiments.CVSMDriver import CVSMDriver
from main.experiments.PRADriver import PRADriver
from main.playground.make_data_format import process_paths
from main.playground.model2.CompositionalVectorAlgorithm import CompositionalVectorAlgorithm

import sys
import os
import shutil
import subprocess
import time
import datetime

# This script is used to run FB15k237 experiments (including baselines) with 1:10 postive to negative ratio.

if __name__ == "__main__":
    # location of Das et al.'s repo
    CVSM_RUN_DIR = "/home/weiyu/Research/Path_Baselines/CVSM/ChainsofReasoning"
    # location of Matt's PRA scala repo
    PRA_RUN_DIR = "/home/weiyu/Research/Path_Baselines/SFE/pra_scala"

    DATASET_NAME = "fb15k237"
    DATASET_FOLDER = os.path.join("data", DATASET_NAME)
    PRA_TEMPLATE_DIR = "pra_templates"

    FREEBASE_DIR = DATASET_FOLDER
    DOMAIN_FILENAME = os.path.join(DATASET_FOLDER, "domains.tsv")
    RANGE_FILENAME = os.path.join(DATASET_FOLDER, "ranges.tsv")
    EDGES_FILENAME = os.path.join(DATASET_FOLDER, "edges.txt")
    PRA_DIR = os.path.join(DATASET_FOLDER, "pra")
    SPLIT_DIR = os.path.join(DATASET_FOLDER, "split")
    CPR_PATH_DIR = os.path.join(DATASET_FOLDER, "cpr_paths")
    WORD2VEC_FILENAME = "data/word2vec/knowledge-vectors-skipgram1000.bin"
    ENTITY2VEC_FILENAME = os.path.join(DATASET_FOLDER, "synonym2vec.pkl")

    # ToDo: modify code to have cvsm_ret, cvsm_r, cvsm_rt, cvsm_re folders (r: relation, t: type, e: entity)
    CVSM_DATA_DIR = os.path.join(DATASET_FOLDER, "cvsm")
    PRA_PATH_DIR = os.path.join(DATASET_FOLDER, "pra_paths")
    RELATION_PATH_DIR = os.path.join(DATASET_FOLDER, "relation_paths")
    PATH_DIR = os.path.join(DATASET_FOLDER, "paths")
    NEW_PATH_DIR = os.path.join(DATASET_FOLDER, "new_paths")
    AUGMENT_PATH_DIR = os.path.join(DATASET_FOLDER, "paths_augment")

    CVSM_RET_DIR = os.path.join(DATASET_FOLDER, "cvsm_entity")
    ENTITY_TYPE2VEC_FILENAME = os.path.join(DATASET_FOLDER, "entity_type2vec.pkl")

    run_step = 13

    # 1. first run main/data/MIDFreebase15kReader.py to process data and generate necessary files
    if run_step == 1:
        fb = MIDFreebase15kReader(FREEBASE_DIR,
                                  filter=True,
                                  word2vec_filename=WORD2VEC_FILENAME)
        fb.read_data()
        fb.write_relation_domain_and_ranges()
        fb.write_edges()

        # get the dictionary from freebase mids to names
        # fb.get_mid_to_name()

        # ToDo: add function to return data statistics that can be used in other functions.

    # 2. use PRA scala code and code here to create train/test/dev split and negative examples
    if run_step == 2:
        pra_driver = PRADriver(DATASET_FOLDER, PRA_TEMPLATE_DIR, PRA_RUN_DIR, DATASET_NAME)
        pra_driver.prepare_split()

    # 3. run PRA and SFE using PRA scala code
    if run_step == 2:
        pra_driver = PRADriver(DATASET_FOLDER, PRA_TEMPLATE_DIR, PRA_RUN_DIR, DATASET_NAME)
        pra_driver.run_pra()

    # Depracated
    # # 2. Generate PRA input files to generate split and negative examples.
    # #    PRA creates split all relation instances arbitrarily based on the train/test ratio. This is different from
    # #    knowledge embedding approaches where train set need to contain entities in test set.
    # if run_step == 2:
    #     typed_relation_instances = TypedRelationInstances()
    #     typed_relation_instances.read_domains_and_ranges(DOMAIN_FILENAME, RANGE_FILENAME)
    #     typed_relation_instances.construct_from_labeled_edges(EDGES_FILENAME, entity_name_is_typed=False, is_labeled=False)
    #     typed_relation_instances.write_to_pra_format(PRA_DIR, only_positive_instance=True)
    #
    # # 3. Run PRA create_graph_and_split, copy generated split to $SPLIT_DIR. remove edge.dat.
    # if run_step == 3:
    #     pass
    #
    # # 4. Create development set
    # if run_step == 4:
    #     typed_relation_instances = TypedRelationInstances()
    #     typed_relation_instances.read_domains_and_ranges(DOMAIN_FILENAME, RANGE_FILENAME)
    #     typed_relation_instances.construct_from_labeled_edges(EDGES_FILENAME, entity_name_is_typed=False, is_labeled=False)
    #     vocabs = Vocabs()
    #     vocabs.build_vocabs(typed_relation_instances)
    #     split = Split()
    #     split.read_splits(SPLIT_DIR, vocabs, entity_name_is_typed=True,
    #                       create_development_set_if_not_exist=True)
    #
    # # 5. Copy new split with development set to PRA and name it "dev_split". Run PRA and SFE with new split.
    # if run_step == 5:
    #     pass
    #
    # # Skip 6-7 because now we are using BFS paths to run all CVSM models.
    # # 6. Extract PRA paths for CVSM. Need to use pra's original split (train/test) because PRA will not extract paths
    # #    for dev.
    # if run_step == 6:
    #     pass

    # 7. Run CVSM (only relation) original main using PRA paths
    if run_step == 7:
        cvsm_driver = CVSMDriver(FREEBASE_DIR, CVSM_RUN_DIR, dataset="freebase", include_entity=False)
        cvsm_driver.run()

    # 8. Extract paths with entities
    if run_step == 8:
        typed_relation_instances = TypedRelationInstances()
        typed_relation_instances.read_domains_and_ranges(DOMAIN_FILENAME, RANGE_FILENAME)
        typed_relation_instances.construct_from_labeled_edges(EDGES_FILENAME, entity_name_is_typed=False,
                                                              is_labeled=False)
        vocabs = Vocabs()
        vocabs.build_vocabs(typed_relation_instances)
        split = Split()
        split.read_splits(SPLIT_DIR, vocabs, entity_name_is_typed=True)
        graph = AdjacencyGraph()
        graph.build_graph(typed_relation_instances, vocabs)

        path_extractor = PathExtractor(max_length=4, include_entity=True, save_dir=PATH_DIR, include_path_len1=True,
                                       max_paths_per_pair=200, multiple_instances_per_pair=False,
                                       max_instances_per_pair=None, paths_sample_method="all_lengths")
        path_extractor.extract_paths(graph, split, vocabs)
        path_extractor.write_paths(split)
        # Note: we can extract path up to 6 but eliminate nodes with large fan-out

    # 9. Extract paths statistics
    if run_step == 9:
        typed_relation_instances = TypedRelationInstances()
        typed_relation_instances.read_domains_and_ranges(DOMAIN_FILENAME, RANGE_FILENAME)
        typed_relation_instances.construct_from_labeled_edges(EDGES_FILENAME, entity_name_is_typed=False,
                                                              is_labeled=False)
        vocabs = Vocabs()
        vocabs.build_vocabs(typed_relation_instances)
        split = Split()
        split.read_splits(SPLIT_DIR, vocabs, entity_name_is_typed=True)
        path_reader = PathReader(save_dir=PATH_DIR)
        path_reader.read_paths(split)

    # 10. Run CVSM (relation + entity + entity type) using paths extracted in step 8
    if run_step == 10:
        cvsm_driver = CVSMDriver(FREEBASE_DIR, CVSM_RUN_DIR, dataset="freebase", include_entity=True, has_entity=True,
                                 augment_data=False, include_entity_type=True)
        cvsm_driver.run()

    # 11. Run CVSM (relation + type) using paths extracted in step 8
    if run_step == 11:
        cvsm_driver = CVSMDriver(FREEBASE_DIR, CVSM_RUN_DIR, dataset="freebase", include_entity=False, has_entity=True,
                                 augment_data=False, include_entity_type=True)
        cvsm_driver.run()

    # 12. Run CVSM (relation) using paths extracted in step 8
    # Important: CVSM (relation) could use paths from pra, sfe, bfs with include_entity set to false,
    #            and bfs with entities. Currently we are using bfs with entities, while leveraging CVSM code to ignore
    #            entities in paths when vectorizing paths. One problem with this method is that repetitions of relation
    #            paths may occur.
    if run_step == 12:
        cvsm_driver = CVSMDriver(FREEBASE_DIR, CVSM_RUN_DIR, dataset="freebase", include_entity=False, has_entity=True,
                                 augment_data=False, include_entity_type=False)
        cvsm_driver.run()

    # 13. Process data for running the model
    if run_step == 13:
        # first convert paths to cvsm format
        cvsm_driver = CVSMDriver(FREEBASE_DIR, CVSM_RUN_DIR, dataset="freebase", include_entity=True, has_entity=True,
                                 augment_data=False, include_entity_type=True)
        cvsm_driver.setup_cvsm_dir()

        # then vectorize cvsm format data
        process_paths(input_dir="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/fb15k237/cvsm_entity/data/data_input",
                      output_dir="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/fb15k237/cvsm_entity/data/data_output",
                      vocab_dir="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/fb15k237/cvsm_entity/data/vocab",
                      isOnlyRelation=False,
                      getOnlyRelation=False,
                      MAX_POSSIBLE_LENGTH_PATH=8,  # the max number of relations in a path + 1
                      NUM_ENTITY_TYPES_SLOTS=7,  # the number of types + 1 (the reason we +1 is to create a meaningless type for all entities)
                      pre_padding=True)

    # 13. Run the model
    # use $tensorboard --logdir runs to see the training progress
    if run_step == 14:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None, attention_method="sat", early_stopping_metric="map")
        # cvsm.train_and_test()

        #cvsm = CompositionalVectorAlgorithm(CVSM_RET_DIR, None, attention_method="specific", early_stopping_metric="map")
        #cvsm.train_and_test()

        #cvsm = CompositionalVectorAlgorithm(CVSM_RET_DIR, None, attention_method="abstract", early_stopping_metric="map")
        #cvsm.train_and_test()

        # Uncomment if need to train only one relation
        # cvsm.train("/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/fb15k237/cvsm_entity/data/data_output/|food|food|nutrients.|food|nutrition_fact|nutrient")

    if run_step == 15:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="lse", attention_method="sat",
                                            early_stopping_metric="map")
        cvsm.train_and_test()

        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="avg", attention_method="sat",
                                            early_stopping_metric="map")
        cvsm.train_and_test()

        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="max", attention_method="sat",
                                            early_stopping_metric="map")
        cvsm.train_and_test()

    if run_step == 16:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="sat", attention_method="sat", early_stopping_metric="map",
                                            visualize=False, calculate_path_attn_stats=False, calculate_type_attn_stats=True,
                                            mid2name_filename="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/fb15k237/mid2name.pkl",
                                            best_models={'|people|person|profession': {'epoch': 10}, '|award|award_winning_work|awards_won.|award|award_honor|award': {'epoch': 16}, '|tv|tv_program|regular_cast.|tv|regular_tv_appearance|actor': {'epoch': 23}, '|music|record_label|artist': {'epoch': 27}, '|award|award_category|nominees.|award|award_nomination|nominated_for': {'epoch': 15}, '|film|film|production_companies': {'epoch': 24}, '|film|film|genre': {'epoch': 13}, '|sports|sports_position|players.|sports|sports_team_roster|team': {'epoch': 25}, '|food|food|nutrients.|food|nutrition_fact|nutrient': {'epoch': 17}, '|education|educational_institution|students_graduates.|education|education|major_field_of_study': {'epoch': 21}})
        cvsm.train_and_test()

    if run_step == 17:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="lse", attention_method="specific",
                                            early_stopping_metric="map")
        cvsm.train_and_test()

########Deprecated#####################################################################################################
    # 1. write data to fit into PRA and generate synonym2vec
    # typed_relation_instances = TypedRelationInstances()
    # typed_relation_instances.read_domains_and_ranges(domain_filename, range_filename)
    # typed_relation_instances.construct_from_labeled_edges(edges_filename, entity_name_is_typed=False, is_labeled=False)
    # typed_relation_instances.write_to_pra_format(pra_dir, only_positive_instance=True)

    # # 2. test cpr
    # typed_relation_instances = TypedRelationInstances()
    # typed_relation_instances.read_domains_and_ranges(domain_filename, range_filename)
    # typed_relation_instances.construct_from_labeled_edges(edges_filename, entity_name_is_typed=False, is_labeled=False)
    # vocabs = Vocabs()
    # vocabs.build_vocabs(typed_relation_instances)
    # graph = AdjacencyGraph()
    # graph.build_graph(typed_relation_instances, vocabs)
    # split = Split()
    # split.read_splits(split_dir, vocabs, entity_name_is_typed=True)
    #
    # # extract cpr paths
    # if not os.path.exists(cpr_path_dir):
    #     context_path_extractor = CPRPathExtractorMP(max_length=6, include_entity=False, number_of_walkers=200,
    #                                               entity2vec_filename=entity2vec_filename, save_dir=cpr_path_dir,
    #                                               include_path_len1=True)
    #     context_path_extractor.extract_paths(graph, split, vocabs)
    #     context_path_extractor.write_paths(split)
    # else:
    #     context_path_extractor = PathReader(save_dir=cpr_path_dir)
    #     context_path_extractor.read_paths(split)
    #
    # pra = PathRankingAlgorithm()
    # pra.train(split, context_path_extractor)
    # pra.test(split, context_path_extractor)

    # 3. test cvsm
    # cvsm_driver = CVSMDriver(wordnet2_dir, cvsm_run_dir)
    # cvsm_driver.run()