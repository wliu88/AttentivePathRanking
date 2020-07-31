import sys, os
import json
import re
import argparse
from collections import defaultdict
import random
import gzip

# Copied from Ras's repo to decouple it with other prepocessing steps.


def process_paths_for_relation(input_dir, out_dir, vocab_dir, isOnlyRelation, getOnlyRelation, MAX_POSSIBLE_LENGTH_PATH,
                               NUM_ENTITY_TYPES_SLOTS, pre_padding):
    entity_type_vocab_file = os.path.join(vocab_dir, "entity_type_vocab.txt")
    relation_vocab_file = os.path.join(vocab_dir, "relation_vocab.txt")
    entity_vocab_file = os.path.join(vocab_dir, "entity_vocab.txt")
    entity_type_map_file = os.path.join(vocab_dir, "entity_to_list_type.json")
    label_vocab_file = os.path.join(vocab_dir, "domain-label")

    if not isOnlyRelation:
        print('reading entity type vocab')
        entity_type_vocab = {}
        with open(entity_type_vocab_file, 'r') as vocab:
            entity_type_vocab = json.load(vocab)
        print('reading entity vocab')
        entity_vocab = {}
        with open(entity_vocab_file, 'r') as vocab:
            entity_vocab = json.load(vocab)
        print('reading entity to type list')
        entity_type_map = {}
        with open(entity_type_map_file, 'r') as f:
            entity_type_map = json.load(f)
    print('reading relation vocab: ' + relation_vocab_file)
    relation_vocab = {}
    with open(relation_vocab_file, 'r') as vocab:
        relation_vocab = json.load(vocab)
    print('Reading label vocab')
    with open(label_vocab_file, 'r') as label_vocab:
        label2int = json.load(label_vocab)

    # Gets the maximum length of paths
    max_length = -1
    train_files = ['/positive_matrix.tsv.translated', '/negative_matrix.tsv.translated', '/dev_matrix.tsv.translated',
                   '/test_matrix.tsv.translated']  ##dont change the ordering or remove entries. This is bad coding, I know.
    for counter, input_file in enumerate(train_files):
        input_file = input_dir + input_file
        print('Processing ' + input_file)
        with open(input_file) as f:
            for entity_count, line in enumerate(f):  # each entity pair
                split = line.split('\t')
                e1 = split[0].strip()
                e2 = split[1].strip()
                paths = split[2].strip()
                split = paths.split('###')
                for path in split:
                    path_len = len(path.split('-'))
                    if not isOnlyRelation:
                        # ent1 - rel1 - ent2 - rel2 - ent3 - #END_RELATION has length 3
                        # path will be rel1-ent2-rel2
                        path_len = int(path_len / 2) + 2
                    if path_len > max_length:
                        max_length = path_len
    print("Max length of all paths are", max_length)
    max_length = min(MAX_POSSIBLE_LENGTH_PATH, max_length)
    print('Max length will be min(max length of all paths, specificed lenght limit):', str(max_length))

    # function input "length" is the maximum number of entity types for an entity
    # this function return a string of ints separated by ","
    def get_entity_types_in_order(entity_types, length):
        assert (length <= len(entity_types))
        type_int_list = []
        for entity_type in entity_types:
            if entity_type in entity_type_vocab:
                type_int_list.append(entity_type_vocab[entity_type])
            else:
                type_int_list.append(entity_type_vocab['#UNK_ENTITY_TYPE'])
        # sort in ascending order
        type_int_list = sorted(type_int_list)
        type_int_list = type_int_list[:length]  # slice of that length
        type_int_list = type_int_list[::-1]  # reverse
        return ','.join(str(i) for i in type_int_list)

    # will be called on data with just relations
    def get_feature_vector_only_relation(relation):
        feature_vector = ''
        # Now add the id for the relation
        if relation in relation_vocab:
            feature_vector = feature_vector + str(relation_vocab[relation])
        else:
            feature_vector = feature_vector + str(relation_vocab['#UNK_RELATION'])
        assert (len(feature_vector.split(' ')) == 1)
        return feature_vector

    def get_feature_vector(prev_entity, relation):
        type_feature_vector = ''
        # get the entity types of the vector
        if prev_entity in entity_type_map:
            entity_types = entity_type_map[prev_entity]
            if len(entity_types) == 0:
                for i in range(
                        NUM_ENTITY_TYPES_SLOTS):  # we dont have type for this entity the feature vector would be all UNKNOWN TYPE token
                    # Weiyu: I replaced #UNK_ENTITY_TYPE with #PAD_TOKEN
                    type_feature_vector = type_feature_vector + str(
                        entity_type_vocab['#PAD_TOKEN']) + ','  # str(entity_type_vocab['#UNK_ENTITY_TYPE']) +','
            else:
                # if len(entity_types) <= NUM_ENTITY_TYPES_SLOTS:
                # create the feature vector
                length = min(NUM_ENTITY_TYPES_SLOTS, len(entity_types))
                extra_padding_length = NUM_ENTITY_TYPES_SLOTS - len(entity_types)
                for i in range(extra_padding_length):
                    type_feature_vector = type_feature_vector + str(entity_type_vocab['#PAD_TOKEN']) + ','
                # Weiyu: I change padding for types from pre to post
                # type_feature_vector = type_feature_vector + get_entity_types_in_order(entity_types, length) + ','
                type_feature_vector = get_entity_types_in_order(entity_types, length) + ',' + type_feature_vector
        else:
            for i in range(
                    NUM_ENTITY_TYPES_SLOTS):  # we dont have type for this entity the feature vector would be all UNKNOWN TYPE token
                # Weiyu: I replaced #UNK_ENTITY_TYPE with #PAD_TOKEN
                type_feature_vector = type_feature_vector + str(
                    entity_type_vocab['#PAD_TOKEN']) + ','  # str(entity_type_vocab['#UNK_ENTITY_TYPE']) +','
        # NEW: add the id for the entity
        if prev_entity in entity_vocab:
            type_feature_vector = type_feature_vector + str(entity_vocab[prev_entity]) + ','
        else:
            try:
                type_feature_vector = type_feature_vector + str(entity_vocab['#UNK_ENTITY']) + ','
            except:
                raise Exception(prev_entity)
        # Now add the id for the relation
        if relation in relation_vocab:
            type_feature_vector = type_feature_vector + str(relation_vocab[relation])
        else:
            type_feature_vector = type_feature_vector + str(relation_vocab['#UNK_RELATION'])

        assert (len(
            type_feature_vector.split(',')) == NUM_ENTITY_TYPES_SLOTS + 2)  # +2 - because of entity and entity_type
        return type_feature_vector

    # Create the feature for PAD token for a path using PAD token of entity, relation, entity type.
    # pad_feature = relation_pad_token (if only using relation
    # pad_feature = type_pad_token, .., type_pad_token, entity_pad_token, relation_pad_token (if not only using relation and

    pad_feature = ''
    if isOnlyRelation or getOnlyRelation:
        pad_feature = str(relation_vocab['#PAD_TOKEN'])
    else:
        for i in range(NUM_ENTITY_TYPES_SLOTS):
            if i == 0:
                pad_feature = pad_feature + str(entity_type_vocab['#PAD_TOKEN'])
            else:
                pad_feature = pad_feature + ',' + str(entity_type_vocab['#PAD_TOKEN'])
        pad_feature = pad_feature + ',' + str(entity_vocab['#PAD_TOKEN'])
        pad_feature = pad_feature + ',' + str(relation_vocab['#PAD_TOKEN'])

    def get_padding(num_pad_features):
        path_feature_vector = ''
        for i in range(num_pad_features):
            if path_feature_vector == '':
                path_feature_vector = pad_feature
            else:
                path_feature_vector = path_feature_vector + ' ' + pad_feature
        return path_feature_vector

    missed_entity_count = 0  # entity pair might be ignored when we are putting constraints on the max length of the path.
    input_files = ['/positive_matrix.tsv.translated', '/negative_matrix.tsv.translated', '/dev_matrix.tsv.translated',
                   '/test_matrix.tsv.translated']
    # clean the directory
    dirs = ['train', 'dev', 'test']
    for directory in dirs:
        output_dir = out_dir + '/' + directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for f in os.listdir(output_dir):
            if os.path.exists(output_dir + '/' + f):
                os.remove(output_dir + '/' + f)

    label = ''
    for input_file_counter, input_file_name in enumerate(input_files):
        if input_file_counter == 0 or input_file_counter == 1:
            output_dir = out_dir + '/train'
            output_file = output_dir + '/' + 'train.txt'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if input_file_counter == 0:
                label = '1'
            if input_file_counter == 1:
                label = '-1'
        if input_file_counter == 2:
            output_dir = out_dir + '/dev'
            output_file = output_dir + '/' + 'dev.txt'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        if input_file_counter == 3:
            output_dir = out_dir + '/test'
            output_file = output_dir + '/' + 'test.txt'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        print('Output dir changed to ' + output_dir)
        input_file = input_dir + input_file_name
        with open(input_file) as f:
            print(input_file)
            for entity_count, line in enumerate(f):  # each entity pair
                split = line.split('\t')
                if len(split) == 4:
                    # only test and dev have label for each entity pair.
                    assert (input_file_counter == 2 or input_file_counter == 3)
                    label = str(split[3].strip())
                e1 = split[0].strip()
                e2 = split[1].strip()
                prev_entity = e1
                # path are seperated by ###
                split = split[2].split('###')
                flag = 0
                # string for all paths between an entity pair
                output_line = ''
                for path_counter, each_path in enumerate(split):
                    prev_entity = e1
                    each_path = each_path.strip()
                    relation_types = each_path.split('-')
                    path_len = len(relation_types)
                    if not isOnlyRelation:
                        path_len = int(path_len/2) + 2  # so 3 months after writing the main, I was wondering why the + 2 - the answer is even if for a one hop path e1 r1 e2 we consider (e1, r1)->(e2, UNK_RELATION); hence path length is atleast 2
                        if getOnlyRelation:
                            path_len = path_len - 1
                    if path_len > max_length:
                        continue
                    # Important: we have the option to pre-pad a path to a specific length or post-pad it
                    # Important: features for steps in a path are seperated by spaces
                    num_pad_features = max_length - path_len
                    # string for a path
                    if not pre_padding:
                        path_feature_vector = str(path_len)
                    else:
                        path_feature_vector = get_padding(num_pad_features)
                    # iterating through every node in the path
                    for token_counter, token in enumerate(relation_types):
                        if not isOnlyRelation:
                            if token_counter % 2 == 0:  # relation
                                relation = token
                                if getOnlyRelation:
                                    type_feature_vector = get_feature_vector_only_relation(relation)
                                else:
                                    type_feature_vector = get_feature_vector(prev_entity, relation)
                                # at the first step of the path
                                if token_counter == 0 and path_feature_vector == '':
                                    path_feature_vector = path_feature_vector + type_feature_vector
                                else:
                                    path_feature_vector = path_feature_vector + ' ' + type_feature_vector
                            else:  # this is an entity
                                prev_entity = token
                        else:
                            relation = token
                            type_feature_vector = get_feature_vector_only_relation(relation)
                            # at the first step of the path
                            if token_counter == 0 and path_feature_vector == '':
                                path_feature_vector = path_feature_vector + type_feature_vector
                            else:
                                path_feature_vector = path_feature_vector + ' ' + type_feature_vector
                    if not isOnlyRelation and not getOnlyRelation:
                        # take care of e2 (target entity) now
                        path_feature_vector = path_feature_vector + ' ' + get_feature_vector(e2, '#END_RELATION')
                    if not pre_padding:
                        if num_pad_features > 0:
                            path_feature_vector = path_feature_vector + ' ' + get_padding(num_pad_features)
                    try:
                        if pre_padding:
                            assert (len(path_feature_vector.split(' ')) == max_length)
                        else:
                            # bc the first value will be the length of the path
                            assert (len(path_feature_vector.split(' ')) - 1 == max_length)
                    except AssertionError:
                        print("Error")
                        print(each_path)
                        print(num_pad_features)
                        print(len(path_feature_vector.split(' ')))
                        print(path_feature_vector.split(' '))
                        print(max_length)
                        print(path_feature_vector)
                        print('===============')
                        # sys.stderr.write("line:\t"+line)
                        # sys.stderr.write("path:\t"+each_path)
                        continue

                    if path_counter == 0 or flag == 0:  # i put the flag check because the first path might be eliminated because it is greater than max_length but path_counter wont be 0
                        flag = 1
                        output_line = output_line + path_feature_vector
                    else:
                        output_line = output_line + ';' + path_feature_vector
                path_counter = len(output_line.split(';'))
                int_label = ''
                int_label = str(label2int['domain'][label.strip()])
                output_line = output_line.strip()
                if len(output_line) == 0:  # this might happen when an entity pair has no paths lesser than length k (eg 3)
                    missed_entity_count = missed_entity_count + 1
                    continue
                output_line = int_label + '\t' + output_line
                output_line = output_line.strip()
                if path_counter != 0:
                    output_file_with_pathlen = output_file + '.' + str(path_counter) + '.int'
                    # with write option a, all entity pairs with same length of paths will be write to the same file.
                    with open(output_file_with_pathlen, 'a') as out:
                        out.write(output_line + '\n')
                if entity_count % 100 == 0:
                    print('Processed ' + str(entity_count) + ' entity pairs')
    print("Missed entity pair count " + str(missed_entity_count))


def process_paths(input_dir, output_dir, vocab_dir, isOnlyRelation, getOnlyRelation, MAX_POSSIBLE_LENGTH_PATH,
                  NUM_ENTITY_TYPES_SLOTS, pre_padding):
    """
    This function triggers another function to vectorize text data.

    input: data and vocabs in CVSM's input format (can be created by write_cvsm_files() in main/features/PathReader.py)
    output: vectorized data

    :param input_dir: path data in cvsm format
    :param output_dir: vectorized path data in cvsm format
    :param vocab_dir: vocab folder in cvsm format
    :param isOnlyRelation: whether paths contain only relations
    :param getOnlyRelation: whether only use relations in paths
    :param MAX_POSSIBLE_LENGTH_PATH: the max number of relations in a path + 1
    :param NUM_ENTITY_TYPES_SLOTS: the max number of types for an entity + 1
                                   (the reason we +1 is to create a meaningless type for all entities)
    :param pre_padding: whether use pre-padding. pre-padding zeros prevents RNN from forgetting
    :return:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        raise Exception("Output directory already exists.")
    for rel in os.listdir(input_dir):
        rel_input_dir = os.path.join(input_dir, rel)
        rel_output_dir = os.path.join(output_dir, rel)
        os.mkdir(rel_output_dir)
        process_paths_for_relation(rel_input_dir, rel_output_dir, vocab_dir, isOnlyRelation, getOnlyRelation,
                                   MAX_POSSIBLE_LENGTH_PATH, NUM_ENTITY_TYPES_SLOTS, pre_padding)


if __name__ == "__main__":
    process_paths(input_dir="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/wordnet3/cvsm_entity/data/data_input",
                  output_dir="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/wordnet3/cvsm_entity/data/data_output",
                  vocab_dir="/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/wordnet3/cvsm_entity/data/vocab",
                  isOnlyRelation=False,
                  getOnlyRelation=False,
                  MAX_POSSIBLE_LENGTH_PATH=5,  # the max number of relations in a path + 1
                  NUM_ENTITY_TYPES_SLOTS=8,  # the number of types + 1 (the reason we +1 is to create a meaningless type for all entities)
                  pre_padding=True)
