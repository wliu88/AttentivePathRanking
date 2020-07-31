import os

# import and build cython
import pyximport
pyximport.install()

from main.data.MIDFreebase15kReader import MIDFreebase15kReader
from main.data.TypedRelationInstances import TypedRelationInstances
from main.data.Vocabs import Vocabs
from main.graphs.AdjacencyGraph import AdjacencyGraph
from main.features.PathExtractor import PathExtractor
from main.data.Split import Split
from main.features.PathReader import PathReader
from main.experiments.CVSMDriver import CVSMDriver
from main.experiments.PRADriver import PRADriver
from main.playground.make_data_format import process_paths
from main.playground.model2.CompositionalVectorAlgorithm import CompositionalVectorAlgorithm

# This script is used to run FB15k237 experiments (only our method) with 1:10 postive to negative ratio.

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

    run_step = 1

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

    # 2. use PRA scala code and code here to create train/test/dev split and negative examples
    if run_step == 2:
        pra_driver = PRADriver(DATASET_FOLDER, PRA_TEMPLATE_DIR, PRA_RUN_DIR, DATASET_NAME)
        pra_driver.prepare_split()

    # 3. Extract paths with entities
    if run_step == 3:
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

    # 4. Extract paths statistics
    if run_step == 4:
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

    # 5. Process data for running the model
    if run_step == 5:
        # first convert paths to cvsm format
        cvsm_driver = CVSMDriver(FREEBASE_DIR, CVSM_RUN_DIR, dataset="freebase", include_entity=True, has_entity=True,
                                 augment_data=False, include_entity_type=True)
        cvsm_driver.setup_cvsm_dir()

        # then vectorize cvsm format data
        process_paths(input_dir=os.path.join(CVSM_RET_DIR, "data/data_input"),
                      output_dir=os.path.join(CVSM_RET_DIR, "data/data_output"),
                      vocab_dir=os.path.join(CVSM_RET_DIR, "data/vocab"),
                      isOnlyRelation=False,
                      getOnlyRelation=False,
                      MAX_POSSIBLE_LENGTH_PATH=8,  # the max number of relations in a path + 1
                      NUM_ENTITY_TYPES_SLOTS=7,  # the number of types + 1 (the reason we +1 is to create a meaningless type for all entities)
                      pre_padding=True)

    # 6. test run of the model
    # use $tensorboard --logdir runs to see the training progress
    if run_step == 6:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None, attention_method="sat", early_stopping_metric="map")
        # cvsm.train_and_test()

        # Uncomment if need to train only one relation
        cvsm.train(os.path.join(CVSM_RET_DIR, "data/data_output/|food|food|nutrients.|food|nutrition_fact|nutrient"))

    # 7. test different path pooling methods
    if run_step == 7:
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

    # 8. visualize type attention
    if run_step == 8:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="sat", attention_method="sat", early_stopping_metric="map",
                                            visualize=False, calculate_path_attn_stats=False, calculate_type_attn_stats=True,
                                            mid2name_filename="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/fb15k237/mid2name.pkl",
                                            best_models={'|people|person|profession': {'epoch': 10}, '|award|award_winning_work|awards_won.|award|award_honor|award': {'epoch': 16}, '|tv|tv_program|regular_cast.|tv|regular_tv_appearance|actor': {'epoch': 23}, '|music|record_label|artist': {'epoch': 27}, '|award|award_category|nominees.|award|award_nomination|nominated_for': {'epoch': 15}, '|film|film|production_companies': {'epoch': 24}, '|film|film|genre': {'epoch': 13}, '|sports|sports_position|players.|sports|sports_team_roster|team': {'epoch': 25}, '|food|food|nutrients.|food|nutrition_fact|nutrient': {'epoch': 17}, '|education|educational_institution|students_graduates.|education|education|major_field_of_study': {'epoch': 21}})
        cvsm.train_and_test()

    # 9. run ablation on type selection
    if run_step == 9:
        cvsm = CompositionalVectorAlgorithm("freebase", CVSM_RET_DIR, None,
                                            pooling_method="lse", attention_method="specific",
                                            early_stopping_metric="map")
        cvsm.train_and_test()