import torch
import torch.nn as nn
import pickle

torch.manual_seed(1)


class FeatureEmbedding(nn.Module):

    def __init__(self, relation_vocab_size, relation_embedding_dim,
                 entity_vocab_size, entity_embedding_dim,
                 entity_type_vocab_size, entity_type_embedding_dim,
                 entity_type_vocab=None, entity_type2vec_filename=None):
        super(FeatureEmbedding, self).__init__()

        self.relation_embeddings = nn.Embedding(relation_vocab_size, relation_embedding_dim).cuda()

        if entity_type2vec_filename is not None and entity_type_vocab is not None:
            self.entity_types_embeddings = None
            self.load_pretrained_entity_types_embeddings(entity_type_vocab, entity_type2vec_filename)
        else:
            for entity_type in entity_type_vocab:
                if entity_type == "#PAD_TOKEN":
                    pad_index = entity_type_vocab[entity_type]
            self.entity_types_embeddings = nn.Embedding(entity_type_vocab_size, entity_type_embedding_dim, padding_idx=pad_index).cuda()

    def load_pretrained_entity_types_embeddings(self, entity_type_vocab, entity_type2vec_filename):
        print("loading entity_type2vec from pickle file:", entity_type2vec_filename)
        entity_type2vec = pickle.load(open(entity_type2vec_filename, "rb"))
        # entity_type2vec doesn't have "#PAD_TOKENS" while entity_type_vocab does
        print(len(entity_type2vec), len(entity_type_vocab))
        assert len(entity_type2vec) + 1 == len(entity_type_vocab)

        entity_type_embedding_dim = 0
        for entity_type in entity_type2vec:
            entity_type_embedding_dim = len(entity_type2vec[entity_type])
            break
        assert entity_type_embedding_dim != 0

        matrix = torch.FloatTensor(len(entity_type_vocab), entity_type_embedding_dim)
        for entity_type in entity_type_vocab:
            index = entity_type_vocab[entity_type]
            if entity_type == "#PAD_TOKEN":
                matrix[index, :] = torch.zeros(1, entity_type_embedding_dim)
            else:
                matrix[index, :] = torch.FloatTensor(entity_type2vec[entity_type])

        # initialize embedding with the matrix. Turn off training
        self.entity_types_embeddings = torch.nn.Embedding.from_pretrained(matrix, freeze=True).cuda()

    def forward(self, x):
        # the input dimension is #paths x #steps x #feats
        # for each feature, num_entity_types type, 1 entity, 1 relation in order
        relation_embeds = self.relation_embeddings(x[:, :, -1])
        types_embeds = self.entity_types_embeddings(x[:, :, :-2])

        return relation_embeds, types_embeds
