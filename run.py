from main.playground.model2.CompositionalVectorAlgorithm import CompositionalVectorAlgorithm


def test_fb():
    cvsm = CompositionalVectorAlgorithm("freebase", "data/fb15k237/cvsm_entity",
                                        entity_type2vec_filename=None,
                                        pooling_method="sat", attention_method="sat", early_stopping_metric="map")
    cvsm.train_and_test()


def test_wn():
    cvsm = CompositionalVectorAlgorithm("wordnet", experiment_dir="data/wn18rr/cvsm_entity",
                                        entity_type2vec_filename="data/wn18rr/entity_type2vec.pkl",
                                        pooling_method="sat", attention_method="sat", early_stopping_metric="map")
    cvsm.train_and_test()


if __name__ == "__main__":
    test_wn()
