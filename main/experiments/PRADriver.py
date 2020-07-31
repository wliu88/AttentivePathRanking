import os
import sys
import shutil
import subprocess
import shlex
import time
import datetime
import json

from main.data.TypedRelationInstances import TypedRelationInstances
from main.data.Vocabs import Vocabs
from main.data.Split import Split


def run_interactive_command(command_dir, command, input=None):
    """
    This function helps run an interactive shell command

    :param command_dir: where the command should executed
    :param command: the command
    :param input: input to the shell command if there is any
    :return:
    """

    cwd = os.getcwd()
    os.chdir(command_dir)
    print("\n\n")
    print("#"*100)
    print("Run command: {}".format(command))

    p = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.STDOUT)
    if input:
        p.communicate(input="{}\n".format(input).encode())
    else:
        p.communicate()

    os.chdir(cwd)
    print("#" * 100, "\n\n")


class PRADriver:
    """
    This class serves as a bridge between this repo and Matt's PRA scala repo
    """

    def __init__(self, data_dir, pra_template_dir, pra_run_dir, dataset):
        """
        Init

        :param data_dir: data folder in this repo
        :param pra_template_dir: where templates for running PRA scala code are stored
        :param pra_run_dir: location of Matt's PRA scala repo
        :param dataset: name of the dataset, can be one of "wn18rr" or "fb15k237"
        """

        self.data_dir = data_dir
        self.pra_run_dir = pra_run_dir
        self.dataset = dataset
        self.pra_template_dir = pra_template_dir
        assert self.dataset in ["wn18rr", "fb15k237"]

    def prepare_split(self):
        """
        This function uses PRA code to generate initial train/test split and negative examples, and then uses code in
        this repo to create train/dev/test split.

        .. note::

            The pra data will be left in /examples folder in the PRA scala repo after running this function.

        .. note::

            PRA creates split all relation instances arbitrarily based on the train/test ratio. This is different from
            knowledge embedding approaches where train set need to contain entities in test set.

        :param run_pra: default False. If set to true, will also run PRA and SFE using the PRA scala code
        :return:
        """
        domain_filename = os.path.join(self.data_dir, "domains.tsv")
        range_filename = os.path.join(self.data_dir, "ranges.tsv")
        edges_filename = os.path.join(self.data_dir, "edges.txt")
        pra_dir = os.path.join(self.data_dir, "pra")
        split_dir = os.path.join(self.data_dir, "split")

        assert not os.path.exists(split_dir), "split folder already exists in {}".format(split_dir)
        pra_template_here = os.path.join(self.pra_template_dir, self.dataset, "examples")
        assert os.path.exists(pra_template_here)
        pra_template_there = os.path.join(self.pra_run_dir, "examples")
        assert not os.path.exists(pra_template_there), "examples folder already exists in {}".format(pra_template_there)

        # 1. Create PRA input files to generate split and negative examples.
        #    PRA creates split all relation instances arbitrarily based on the train/test ratio. This is different from
        #    knowledge embedding approaches where train set need to contain entities in test set.
        typed_relation_instances = TypedRelationInstances()
        typed_relation_instances.read_domains_and_ranges(domain_filename, range_filename)
        typed_relation_instances.construct_from_labeled_edges(edges_filename, entity_name_is_typed=False,
                                                              is_labeled=False)
        typed_relation_instances.write_to_pra_format(pra_dir, only_positive_instance=True)

        # 2. Run PRA create_graph_and_split, copy generated split to $SPLIT_DIR.
        # copy the template folder to PRA scala repo
        shutil.copytree(pra_template_here, pra_template_there)

        # copy the data to the PRA scala repo
        relation_data_here = pra_dir
        relation_data_there = os.path.join(pra_template_there, "relation_metadata",
                                           {"wn18rr": "wordnet", "fb15k237": "freebase"}[self.dataset])
        if os.path.exists(relation_data_there):
            shutil.rmtree(relation_data_there)
        shutil.copytree(relation_data_here, relation_data_there)

        # run create_graph_and_split
        command = "sbt \"run ./examples/ {}_create_graph_and_split.json\"".format(
            {"wn18rr": "wordnet", "fb15k237": "freebase"}[self.dataset])
        run_interactive_command(self.pra_run_dir, command, input=1)

        # remove edge.dat
        os.remove(os.path.join(pra_template_there, "graphs",
                               {"wn18rr": "wordnet", "fb15k237": "freebase"}[self.dataset], "edges.dat"))

        # copy generated split to this repo
        split_dir_here = split_dir
        split_dir_there = os.path.join(pra_template_there, "splits/split")
        shutil.copytree(split_dir_there, split_dir_here)

        # 3. Create development set
        typed_relation_instances = TypedRelationInstances()
        typed_relation_instances.read_domains_and_ranges(domain_filename, range_filename)
        typed_relation_instances.construct_from_labeled_edges(edges_filename, entity_name_is_typed=False,
                                                              is_labeled=False)
        vocabs = Vocabs()
        vocabs.build_vocabs(typed_relation_instances)
        split = Split()
        split.read_splits(split_dir, vocabs, entity_name_is_typed=True,
                          create_development_set_if_not_exist=True)

        # 4. Copy new split to PRA scala repo
        split_dir_here = split_dir
        split_dir_there = os.path.join(pra_template_there, "splits/dev_split")
        shutil.copytree(split_dir_here, split_dir_there)

    def run_pra(self):
        """
        This function runs PRA algorithm and SFE algorithm

        :return:
        """
        pra_template_there = os.path.join(self.pra_run_dir, "examples")
        assert os.path.exists(pra_template_there), "examples folder does not exist in {}".format(pra_template_there)

        command = "sbt \"run ./examples/ {}_pra.json\"".format(
            {"wn18rr": "wordnet", "fb15k237": "freebase"}[self.dataset])
        run_interactive_command(self.pra_run_dir, command, input=1)
        run_interactive_command(self.pra_run_dir, command, input=2)

        command = "sbt \"run ./examples/ {}_sfe.json\"".format(
            {"wn18rr": "wordnet", "fb15k237": "freebase"}[self.dataset])
        run_interactive_command(self.pra_run_dir, command, input=1)
        run_interactive_command(self.pra_run_dir, command, input=2)
