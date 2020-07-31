# import and build cython
import pyximport
pyximport.install()

# need to test if predicted relation instance is in paths.
import unittest
import shutil
from main.features.PathExtractor import PathExtractor


class TestVocabs:
    def __init__(self):
        self.idx_to_node = {1: "ent1", 2: "ent2", 3: "ent3", 4: "ent4", 5: "ent5", 6: "ent6", 7: "ent7"}
        self.node_to_idx = {"ent1": 1, "ent2": 2, "ent3": 3, "ent4": 4, "ent5": 5, "ent6": 6, "ent7": 7}
        self.idx_to_relation = {1: "rel1", 2: "rel2", 3: "rel3", 4: "_rel1", 5: "_rel2", 6: "_rel3"}
        self.relation_to_idx = {"rel1": 1, "rel2": 2, "rel3": 3, "_rel1": 4, "_rel2": 5, "_rel3": 6}
        self.idx_to_rev_relation_idx = {1: 4, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3}


class TestGraph:
    def __init__(self):
        self.node_to_parents = {}
        self.node_to_children = {}
        self.pair_to_relations = {}

    def build_graph(self, relation_instances, vocabs):
        for subj, rel, obj in relation_instances:
            # forward direction source ->edge-> target
            source = vocabs.node_to_idx[subj]
            target = vocabs.node_to_idx[obj]
            edge = vocabs.relation_to_idx[rel]
            if source not in self.node_to_children:
                self.node_to_children[source] = set()
            self.node_to_children[source].add(target)
            if target not in self.node_to_parents:
                self.node_to_parents[target] = set()
            self.node_to_parents[target].add(source)
            if (source, target) not in self.pair_to_relations:
                self.pair_to_relations[(source, target)] = set()
            self.pair_to_relations[(source, target)].add(edge)

            # reverse direction target ->rev_edge-> source
            if rel[0] == "-":
                rev_edge = vocabs.relation_to_idx[rel[1:]]
            else:
                rev_edge = vocabs.relation_to_idx["_" + rel]
            if (target, source) not in self.pair_to_relations:
                self.pair_to_relations[(target, source)] = set()
            self.pair_to_relations[(target, source)].add(rev_edge)


simple_edges_list = [("ent1", "rel1", "ent2"), ("ent2", "rel1", "ent3"), ("ent3", "rel1", "ent4"),
                     ("ent4", "rel1", "ent5"), ("ent5", "rel1", "ent6"), ("ent6", "rel1", "ent7")]


class TestPathExtractors(unittest.TestCase):
    def setUp(self):
        pass

    def test_path_to_string1(self):
        path_extractor = PathExtractor(max_length=4, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        assert "ent1#rel1#ent2" == path_extractor.path_to_string((1, 1, 2), vocabs)
        assert "ent1#rel1#" == path_extractor.path_to_string((1, 1, 2), vocabs, drop_last=True)
        assert "ent2#_rel1#ent1" == path_extractor.path_to_string((1, 1, 2), vocabs, reverse=True)
        shutil.rmtree("test_data")

    def test_path_to_string2(self):
        path_extractor = PathExtractor(max_length=4, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        assert "rel1" == path_extractor.path_to_string((1, 1, 2), vocabs)
        assert "rel1#rel2" == path_extractor.path_to_string((1, 1, 2, 2, 3), vocabs)
        assert "_rel1" == path_extractor.path_to_string((1, 1, 2), vocabs, reverse=True)
        assert "_rel2#_rel1" == path_extractor.path_to_string((1, 1, 2, 2, 3), vocabs, reverse=True)
        shutil.rmtree("test_data")

    def test_extract_paths1a(self):
        path_extractor = PathExtractor(max_length=2, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph(simple_edges_list, vocabs)

        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 0
        paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        assert len(paths) == 0
        paths = path_extractor.get_paths("ent2", "ent6", graph, vocabs)
        assert len(paths) == 0
        paths = path_extractor.get_paths("ent1", "ent4", graph, vocabs)
        assert len(paths) == 0
        paths = path_extractor.get_paths("ent3", "ent6", graph, vocabs)
        assert len(paths) == 0

    def test_extract_paths1a(self):
        path_extractor = PathExtractor(max_length=2, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph(simple_edges_list, vocabs)

        for num1 in range(1, 8):
            for num2 in range(1, 8):
                if abs(num1 - num2) <= 1 or abs(num1 - num2) > 2:
                    paths = path_extractor.get_paths("ent"+str(num1), "ent"+str(num2), graph, vocabs)
                    assert len(paths) == 0
                else:
                    paths = path_extractor.get_paths("ent" + str(num1), "ent" + str(num2), graph, vocabs)
                    if num2 > num1:
                        assert len(paths) == 1 and paths.pop() == "ent"+str(num1)+"#rel1#ent"+str(num1+1)+"#rel1#ent"+str(num2)
                    else:
                        assert len(paths) == 1 and paths.pop() == "ent" + str(num1) + "#_rel1#ent" + str(num2 + 1) + "#_rel1#ent" + str(num2)

    # def test_extract_paths1b(self):
    #     path_extractor = PathExtractor(max_length=2, include_entity=False, save_dir="test_data")
    #     vocabs = TestVocabs()
    #     graph = TestGraph()
    #     graph.build_graph(simple_edges_list, vocabs)
    #
    #     for num1 in range(1, 8):
    #         for num2 in range(1, 8):
    #             if abs(num1 - num2) <= 1 or abs(num1 - num2) > 2:
    #                 paths = path_extractor.get_paths("ent" + str(num1), "ent" + str(num2), graph, vocabs)
    #                 assert len(paths) == 0
    #             else:
    #                 paths = path_extractor.get_paths("ent" + str(num1), "ent" + str(num2), graph, vocabs)
    #                 if num2 > num1:
    #                     assert len(paths) == 1 and paths.pop() == "rel1#rel1"
    #                 else:
    #                     assert len(paths) == 1 and paths.pop() == "_rel1#_rel1"




        # paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        # assert len(paths) == 0
        # paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        # assert len(paths) == 0
        # paths = path_extractor.get_paths("ent2", "ent7", graph, vocabs)
        # assert len(paths) == 0
        # paths = path_extractor.get_paths("ent3", "ent5", graph, vocabs)
        # assert len(paths) == 1 and paths.pop() == "ent3#rel1#ent4#rel1#ent5"
        # paths = path_extractor.get_paths("ent6", "ent4", graph, vocabs)
        # assert len(paths) == 1 and paths.pop() == "ent3#rel1#ent4#rel1#ent5"

    def test_extract_paths1b(self):
        path_extractor = PathExtractor(max_length=6, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel1", "ent3"), ("ent3", "rel1", "ent4"),
                           ("ent4", "rel1", "ent5"), ("ent5", "rel1", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1 and paths.pop() == "ent1#rel1#ent2#rel1#ent3#rel1#ent4#rel1#ent5#rel1#ent6#rel1#ent7"
        paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        assert len(paths) == 1 and paths.pop() == "ent7#_rel1#ent6#_rel1#ent5#_rel1#ent4#_rel1#ent3#_rel1#ent2#_rel1#ent1"
        paths = path_extractor.get_paths("ent6", "ent2", graph, vocabs)
        assert len(paths) == 1 and paths.pop() == "ent6#_rel1#ent5#_rel1#ent4#_rel1#ent3#_rel1#ent2"

    def test_extract_paths1c(self):
        path_extractor = PathExtractor(max_length=8, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel1", "ent3"), ("ent3", "rel1", "ent4"),
                           ("ent4", "rel1", "ent5"), ("ent5", "rel1", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1 and paths.pop() == "ent1#rel1#ent2#rel1#ent3#rel1#ent4#rel1#ent5#rel1#ent6#rel1#ent7"
        paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        assert len(paths) == 1 and paths.pop() == "ent7#_rel1#ent6#_rel1#ent5#_rel1#ent4#_rel1#ent3#_rel1#ent2#_rel1#ent1"
        paths = path_extractor.get_paths("ent6", "ent2", graph, vocabs)
        assert len(paths) == 1 and paths.pop() == "ent6#_rel1#ent5#_rel1#ent4#_rel1#ent3#_rel1#ent2"


    def test_extract_paths1b(self):
        path_extractor = PathExtractor(max_length=4, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","rel1","ent3"), ("ent3","rel1","ent4"),
                           ("ent4","rel1","ent5"), ("ent5","rel1","ent6"), ("ent6","rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 0
        paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        assert len(paths) == 0

        path_extractor = PathExtractor(max_length=6, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel1", "ent3"), ("ent3", "rel1", "ent4"),
                           ("ent4", "rel1", "ent5"), ("ent5", "rel1", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "rel1#rel1#rel1#rel1#rel1#rel1"
        paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "_rel1#_rel1#_rel1#_rel1#_rel1#_rel1"

        path_extractor = PathExtractor(max_length=8, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel1", "ent3"), ("ent3", "rel1", "ent4"),
                           ("ent4", "rel1", "ent5"), ("ent5", "rel1", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "rel1#rel1#rel1#rel1#rel1#rel1"
        paths = path_extractor.get_paths("ent7", "ent1", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "_rel1#_rel1#_rel1#_rel1#_rel1#_rel1"

    def test_extract_paths2a(self):
        path_extractor = PathExtractor(max_length=4, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 0

        path_extractor = PathExtractor(max_length=6, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel2", "ent3"), ("ent3", "rel3", "ent4"),
                           ("ent4", "rel3", "ent5"), ("ent5", "rel2", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "ent1#rel1#ent2#rel2#ent3#rel3#ent4#rel3#ent5#rel2#ent6#rel1#ent7"

        path_extractor = PathExtractor(max_length=8, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel2", "ent3"), ("ent3", "rel3", "ent4"),
                           ("ent4", "rel3", "ent5"), ("ent5", "rel2", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "ent1#rel1#ent2#rel2#ent3#rel3#ent4#rel3#ent5#rel2#ent6#rel1#ent7"

    def test_extract_paths2b(self):
        path_extractor = PathExtractor(max_length=4, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 0

        path_extractor = PathExtractor(max_length=6, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel2", "ent3"), ("ent3", "rel3", "ent4"),
                           ("ent4", "rel3", "ent5"), ("ent5", "rel2", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "rel1#rel2#rel3#rel3#rel2#rel1"

        path_extractor = PathExtractor(max_length=8, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1", "rel1", "ent2"), ("ent2", "rel2", "ent3"), ("ent3", "rel3", "ent4"),
                           ("ent4", "rel3", "ent5"), ("ent5", "rel2", "ent6"), ("ent6", "rel1", "ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "rel1#rel2#rel3#rel3#rel2#rel1"

















    def test_extract_paths3a(self):
        path_extractor = PathExtractor(max_length=4, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","_rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","_rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","_rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 0

        path_extractor = PathExtractor(max_length=6, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","_rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","_rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","_rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "ent1#rel1#ent2#rel2#ent3#rel3#ent4#rel3#ent5#rel2#ent6#rel1#ent7"

        path_extractor = PathExtractor(max_length=8, include_entity=True, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","_rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","_rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","_rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "ent1#rel1#ent2#rel2#ent3#rel3#ent4#rel3#ent5#rel2#ent6#rel1#ent7"

    def test_extract_paths3b(self):
        path_extractor = PathExtractor(max_length=4, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","_rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","_rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","_rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 0

        path_extractor = PathExtractor(max_length=6, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","_rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","_rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","_rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "rel1#rel2#rel3#rel3#rel2#rel1"

        path_extractor = PathExtractor(max_length=8, include_entity=False, save_dir="test_data")
        vocabs = TestVocabs()
        graph = TestGraph()
        graph.build_graph([("ent1","rel1","ent2"), ("ent2","_rel2","ent3"), ("ent3","rel3","ent4"),
                           ("ent4","_rel3","ent5"), ("ent5","rel2","ent6"), ("ent6","_rel1","ent7")], vocabs)
        paths = path_extractor.get_paths("ent1", "ent7", graph, vocabs)
        assert len(paths) == 1
        for path in paths:
            assert path == "rel1#rel2#rel3#rel3#rel2#rel1"







