from gensim.models import KeyedVectors
from nltk.corpus import wordnet
import tqdm

# the following four lines is an example of changing glove to word2vec format
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove_input_file = 'glove.6B/glove.6B.100d.txt'
# word2vec_output_file = 'glove.6B/glove.6B.100d.txt.word2vec'
# glove2word2vec(glove_input_file, word2vec_output_file)


# class Word2Vec:
#     def __init__(self, filename):
#         self.word2vec_filename = filename
#         self.model = KeyedVectors.load_word2vec_format(self.word2vec_filename, binary=False)
#
#     def get_most_similar(self, positive, negative, topn):
#
# result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(result)

if __name__ == "__main__":
    entities = set()
    fail_entities = set()
    edges_filename = "/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/wordnet/edges.txt"
    with open(edges_filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if len(line) == 0:
                continue
            subject, rel, object = line.split("\t")
            entities.add(subject)
            entities.add(object)

    filename = "/home/weiyu/Research/ChainsOfReasoningWithAbstractEntities/data/word2vec/GoogleNews-vectors-negative300.bin"
    model = KeyedVectors.load_word2vec_format(filename, binary=True)
    for entity in entities:
        fail = False
        word = entity.split(".")[0]
        # print("\n")
        try:
            model.get_vector(word)
            result = model.most_similar(positive=[word], topn=5)
            # print(entity, wordnet.synset(entity), result)
        except KeyError:
            # print(entity, "not in word2vec")
            if "_" in word:
                # google news may expect each word in a compound word to be capitalized
                transform_word = "_".join([w.capitalize() for w in word.split("_")])
                try:
                    model.get_vector(transform_word)
                    result = model.most_similar(positive=[transform_word], topn=5)
                    # print("transform works", transform_word, wordnet.synset(entity), result)
                except KeyError:
                    pass
                    # print("transform fails", transform_word)
            # print("try synonyms", wordnet.synset(entity).lemma_names())
            for synonym in wordnet.synset(entity).lemma_names():
                try:
                    model.get_vector(synonym)
                    result = model.most_similar(positive=[synonym], topn=5)
                    # print("synonym works", synonym, wordnet.synset(entity), result)
                    break
                except KeyError:
                    fail = True
        if fail is True:
            fail_entities.add(entity)

    print(fail_entities)
    print(len(fail_entities), "fails out of", len(entities))




