import unittest
import shutil
import os
from main.data.TypedRelationInstances import TypedRelationInstances


class TypedRelationInstancesTest(unittest.TestCase):
    def setUp(self):
        self.dir = "test_data"
        os.mkdir(self.dir)
        self.domain_file = os.path.join(self.dir, "domain.txt")
        self.range_file = os.path.join(self.dir, "range.txt")
        self.edges_file = os.path.join(self.dir, "edges.txt")
        self.typed_relation_instances = TypedRelationInstances()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_domain_and_range(self):
        # object in location
        # object made_of material
        with open(self.domain_file, "w+") as fh:
            fh.writelines(["in\tobject\n", "made_of\tobject\n"])
        with open(self.range_file, "w+") as fh:
            fh.writelines(["in\tlocation\n", "made_of\tmaterial\n"])
        self.typed_relation_instances.read_domains_and_ranges(self.domain_file, self.range_file)
        assert "object" == self.typed_relation_instances.relation_domain["in"]
        assert "object" == self.typed_relation_instances.relation_domain["made_of"]
        assert "location" == self.typed_relation_instances.relation_range["in"]
        assert "material" == self.typed_relation_instances.relation_range["made_of"]
        assert len(self.typed_relation_instances.relation_range) == 2
        assert len(self.typed_relation_instances.relation_domain) == 2

    def test_simple_edges(self):
        # object in location
        # object made_of material
        with open(self.domain_file, "w+") as fh:
            fh.writelines(["in\tobject\n", "made_of\tobject\n"])
        with open(self.range_file, "w+") as fh:
            fh.writelines(["in\tlocation\n", "made_of\tmaterial\n"])
        self.typed_relation_instances.read_domains_and_ranges(self.domain_file, self.range_file)

        # apple in basket
        # watermelon in refrigerator
        # water made_of water
        with open(self.edges_file, "w+") as fh:
            fh.write("apple\tin\tbasket\nwatermelon\tin\trefrigerator\nwater\tmade_of\twater\n")
        self.typed_relation_instances.construct_from_labeled_edges(self.edges_file, entity_name_is_typed=False, is_labeled=False)
        assert len(self.typed_relation_instances.type_to_entities) == 3
        assert self.typed_relation_instances.type_to_entities["object"] == {"object:apple", "object:watermelon",
                                                                            "object:water"}
        assert self.typed_relation_instances.type_to_entities["location"] == {"location:basket", "location:refrigerator"}
        assert self.typed_relation_instances.type_to_entities["material"] == {"material:water"}


