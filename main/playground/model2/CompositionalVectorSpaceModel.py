import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import collections
import os
import random
import time
import numpy as np
import json

from main.playground.model2.FeatureEmbedding import FeatureEmbedding

torch.manual_seed(1)


def print_sum(module, grad_input, grad_output):
    return print(grad_output[0].flatten().sum())


class RelationEncoder(nn.Module):
    def __init__(self, relation_embedding_dim, rnn_hidden_dim):
        super(RelationEncoder, self).__init__()

        self.rnn_hidden_dim = rnn_hidden_dim
        self.lstm = nn.LSTM(relation_embedding_dim, rnn_hidden_dim, batch_first=True).cuda()

    def init_hidden(self, batch_size):
        # Hidden state axes semantics are (seq_len, batch, rnn_hidden_dim), even when LSTM is set to batch first
        hidden_state = torch.cuda.FloatTensor(1, batch_size, self.rnn_hidden_dim)
        hidden_state.copy_(torch.zeros(1, batch_size, self.rnn_hidden_dim))
        cell_state = torch.cuda.FloatTensor(1, batch_size, self.rnn_hidden_dim)
        cell_state.copy_(torch.zeros(1, batch_size, self.rnn_hidden_dim))
        return (hidden_state, cell_state)

    def forward(self, relation_embeds):
        # relation_embeds: [num_ent_pairs x num_paths, num_steps, num_feats]
        reshaped_batch_size, num_steps, num_feats = relation_embeds.shape

        _, (last_hidden, _) = self.lstm(relation_embeds, self.init_hidden(reshaped_batch_size))
        last_hidden = last_hidden.squeeze(dim=0)
        # last_hidden: [num_ent_pairs x num_paths, rnn_hidden_dim]
        return last_hidden


class Attention(nn.Module):

    def __init__(self, types_embedding_dim, full_encoder_dim, attention_dim, attention_method="sat"):
        super(Attention, self).__init__()
        self.attention_method = attention_method
        if self.attention_method == "sat":
            self.type_encoder_att = nn.Linear(types_embedding_dim, attention_dim).cuda()
            self.full_encoder_att = nn.Linear(full_encoder_dim, attention_dim).cuda()
            self.full_att = nn.Linear(attention_dim, 1).cuda()
            self.relu = nn.ReLU().cuda()
            self.softmax = nn.Softmax(dim=1).cuda()
        elif self.attention_method == "general":
            self.full_encoder_dim = full_encoder_dim
            self.linear_in = nn.Linear(types_embedding_dim, full_encoder_dim, bias=False).cuda()
            self.softmax = nn.Softmax(dim=1).cuda()
        elif self.attention_method == "abstract" or self.attention_method == "specific" or self.attention_method == "random":
            self.type_encoder_att = nn.Linear(types_embedding_dim, attention_dim).cuda()

    def forward(self, types_embeds, full_encoder_hidden):

        if self.attention_method == "abstract":
            reshaped_batch_size, num_types, _ = types_embeds.shape
            types_embeds = self.type_encoder_att(types_embeds)
            attention_weighted_type_embeds = types_embeds[:, -1, :]
            alpha = torch.cuda.FloatTensor(reshaped_batch_size, num_types).fill_(0)
            alpha[:, -1] = 1.0
        elif self.attention_method == "specific":
            reshaped_batch_size, num_types, _ = types_embeds.shape
            types_embeds = self.type_encoder_att(types_embeds)
            attention_weighted_type_embeds = types_embeds[:, 0, :]
            alpha = torch.cuda.FloatTensor(reshaped_batch_size, num_types).fill_(0)
            alpha[:, 0] = 1.0
        elif self.attention_method == "random":
            reshaped_batch_size, num_types, types_embedding_dim = types_embeds.shape
            types_embeds = self.type_encoder_att(types_embeds)
            dim1 = torch.cuda.LongTensor(list(range(reshaped_batch_size)))
            dim2 = torch.cuda.LongTensor(np.random.randint(0, num_types, size=reshaped_batch_size))
            attention_weighted_type_embeds = types_embeds[dim1, dim2, :]
            alpha = torch.cuda.FloatTensor(reshaped_batch_size, num_types).fill_(0)
            alpha[dim1, dim2] = 1.0
        elif self.attention_method == "sat":
            # type_embeds: [num_ent_pairs x num_paths, num_types, type_encoder_dim]
            att1 = self.type_encoder_att(types_embeds)
            # full_encoder_hidden: [num_ent_pairs x num_paths, full_encoder_dim]
            att2 = self.full_encoder_att(full_encoder_hidden)
            att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
            # att: [num_ent_pairs x num_paths, num_types]
            alpha = self.softmax(att)
            attention_weighted_type_embeds = (att1 * alpha.unsqueeze(2)).sum(dim=1)
        elif self.attention_method == "general":
            # type_embeds: [num_ent_pairs x num_paths, num_types, type_encoder_dim]
            # full_encoder_hidden: [num_ent_pairs x num_paths, full_encoder_dim]
            context = self.linear_in(types_embeds)
            # context: [num_ent_pairs x num_paths, num_types, full_encoder_dim]
            full_encoder_hidden = full_encoder_hidden.unsqueeze(dim=1)
            # full_encoder_hidden: [num_ent_pairs x num_paths, 1, full_encoder_dim]
            attention_scores = torch.matmul(full_encoder_hidden, context.transpose(1, 2).contiguous())
            # attention_scores: [num_ent_pairs x num_paths, 1, num_types]
            alpha = self.softmax(attention_scores.squeeze(dim=1))
            attention_weighted_type_embeds = (types_embeds * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_type_embeds, alpha


class CompositionalVectorSpaceModel(nn.Module):

    def __init__(self, relation_vocab_size, entity_vocab_size, entity_type_vocab_size,
                 relation_embedding_dim, entity_embedding_dim, entity_type_embedding_dim,
                 entity_type_vocab, entity_type2vec_filename,
                 attention_dim, relation_encoder_dim, full_encoder_dim,
                 pooling_method="sat", attention_method="sat"):

        super(CompositionalVectorSpaceModel, self).__init__()

        # params
        # relation_vocab_size = relation_vocab_size
        # relation_embedding_dim = relation_embedding_dim # 250
        # entity_vocab_size = entity_vocab_size
        # entity_embedding_dim = entity_embedding_dim
        # entity_type_vocab_size = entity_type_vocab_size
        # entity_type_embedding_dim = entity_type_embedding_dim
        label_dim = 1

        # Networks
        self.feature_embeddings = FeatureEmbedding(relation_vocab_size, relation_embedding_dim,
                                                   entity_vocab_size, entity_embedding_dim,
                                                   entity_type_vocab_size, entity_type_embedding_dim,
                                                   entity_type_vocab, entity_type2vec_filename)

        self.relation_encoder = RelationEncoder(relation_embedding_dim, relation_encoder_dim)

        self.attention = Attention(entity_type_embedding_dim, full_encoder_dim, attention_dim,
                                   attention_method=attention_method)

        self.full_encoder_step = nn.LSTMCell(attention_dim, full_encoder_dim).cuda()

        # predict initial state for second encoder
        self.init_h = nn.Linear(relation_encoder_dim, full_encoder_dim).cuda()
        self.init_c = nn.Linear(relation_encoder_dim, full_encoder_dim).cuda()

        # attention gate
        self.f_beta = nn.Linear(full_encoder_dim, attention_dim).cuda()

        self.sigmoid = nn.Sigmoid().cuda()

        self.pooling_method = pooling_method
        if self.pooling_method == "lse":
            self.fc = nn.Linear(full_encoder_dim + relation_encoder_dim, label_dim).cuda()
        elif self.pooling_method == "hat":
            path_hidden_dim = 100
            self.path_projector = nn.Linear(full_encoder_dim, path_hidden_dim).cuda()
            self.tanh = nn.Tanh().cuda()
            self.path_context = nn.Parameter(torch.cuda.FloatTensor(path_hidden_dim))
            torch.nn.init.normal_(self.path_context)
            self.softmax = nn.Softmax(dim=1).cuda()
            self.fc = nn.Linear(full_encoder_dim, label_dim).cuda()
        elif self.pooling_method == "sat":
            path_hidden_dim = 100
            self.path_context = nn.Parameter(torch.cuda.FloatTensor(path_hidden_dim))
            torch.nn.init.normal_(self.path_context)
            self.path_att = nn.Linear(full_encoder_dim + relation_encoder_dim, path_hidden_dim).cuda()
            self.att = nn.Linear(path_hidden_dim, 1).cuda()
            self.relu = nn.ReLU().cuda()
            self.softmax = nn.Softmax(dim=1).cuda()
            self.fc = nn.Linear(full_encoder_dim + relation_encoder_dim, label_dim).cuda()
            # self.dropout = nn.Dropout(p=0.5)
        elif self.pooling_method == "max":
            self.fc = nn.Linear(full_encoder_dim + relation_encoder_dim, label_dim).cuda()
        elif self.pooling_method == "avg":
            self.fc = nn.Linear(full_encoder_dim + relation_encoder_dim, label_dim).cuda()

    def init_hidden(self, relation_encoder_out):
        # relation_encoder_out: [num_ent_pairs x num_paths, relation_encoder_dim]
        h = self.init_h(relation_encoder_out)
        c = self.init_c(relation_encoder_out)
        return h, c

    def forward(self, x):
        # x: [num_ent_pairs, num_paths, num_steps, num_feats]
        num_ent_pairs, num_paths, num_steps, num_feats = x.shape
        # collide dim 0 and dim 1
        reshaped_batch_size = num_ent_pairs * num_paths
        x = x.view(reshaped_batch_size, num_steps, num_feats)
        # x: [num_ent_pairs x num_paths, num_steps, num_feats]

        relation_embeds, types_embeds = self.feature_embeddings(x)
        # relation_embeds: [num_ent_pairs x num_paths, num_steps, relation_embedding_dim]
        # types_embeds: [num_ent_pairs x num_paths, num_steps, num_types, entity_type_embedding_dim]

        relation_encoder_out = self.relation_encoder(relation_embeds)
        # relation_encoder_out: [num_ent_pairs x num_paths, relation_encoder_dim]

        h, c = self.init_hidden(relation_encoder_out)
        # h or c: [num_ent_pairs x num_paths, full_encoder_dim]

        num_types = types_embeds.shape[2]
        alphas = torch.cuda.FloatTensor(reshaped_batch_size, num_steps, num_types)
        for t in range(num_steps):
            types_embeds_t = types_embeds[:, t, :, :]
            # types_embeds_t: [num_ent_pairs x num_paths, num_types, entity_type_embedding_dim]
            attention_weighted_encoding, alpha = self.attention(types_embeds_t, h)
            # alpha: [num_ent_pairs x num_paths, num_types]
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            # attention_weighted_encoding: [num_ent_pairs x num_paths, entity_type_embedding_dim]

            feats_t = attention_weighted_encoding

            h, c = self.full_encoder_step(feats_t, (h, c))
            alphas[:, t, :] = alpha

        h = torch.cat((h, relation_encoder_out), dim=1)

        path_weights = torch.cuda.FloatTensor(num_ent_pairs, num_paths)
        if self.pooling_method == "lse":
            path_scores = self.fc(h)
            # path_scores: [num_ent_pairs x num_paths, label_dim]
            path_scores = path_scores.view(num_ent_pairs, num_paths, -1)
            # path_scores: [num_ent_pairs, num_paths, label_dim]
            # LogSumExp
            maxes, max_indices = torch.max(path_scores, dim=1, keepdim=True)
            # print(maxes.squeeze())
            score_minus_maxes = torch.add(path_scores, -1, maxes.expand_as(path_scores))
            exp_score_minus_max = torch.exp(score_minus_maxes)
            sum_exp_score_minus_max = torch.sum(exp_score_minus_max, dim=1)
            lse_scores = torch.log(sum_exp_score_minus_max)
            lse_scores = lse_scores + maxes.squeeze(dim=2)
            # print("lse scores shape", lse_scores.shape)
            # print("maxes shape", maxes.shape)
            probs = self.sigmoid(lse_scores).squeeze(dim=1)
            # probs: [num_ent_pairs, 1]
        elif self.pooling_method == "max":
            path_scores = self.fc(h)
            # path_scores: [num_ent_pairs x num_paths, label_dim]
            path_scores = path_scores.view(num_ent_pairs, num_paths, -1)
            # path_scores: [num_ent_pairs, num_paths, label_dim]
            max_path_score, _ = torch.max(path_scores, dim=1)
            probs = self.sigmoid(max_path_score).squeeze(dim=1)
        elif self.pooling_method == "avg":
            path_scores = self.fc(h)
            # path_scores: [num_ent_pairs x num_paths, label_dim]
            path_scores = path_scores.view(num_ent_pairs, num_paths, -1)
            # path_scores: [num_ent_pairs, num_paths, label_dim]
            path_score_sum = torch.sum(path_scores, dim=1)
            probs = self.sigmoid(path_score_sum).squeeze(dim=1)
        elif self.pooling_method == "hat":
            # h: [num_ent_pairs x num_paths, full_encoder_dim]
            paths_projected = self.tanh(self.path_projector(h))
            path_sims = paths_projected.matmul(self.path_context)
            path_sims = path_sims.view(num_ent_pairs, num_paths, -1)
            path_weights = self.softmax(path_sims)
            # path_weights: [num_ent_pairs, num_paths, 1]
            paths_feats = h.view(num_ent_pairs, num_paths, -1)
            paths_weighted_sum = (paths_feats * path_weights).sum(dim=1)
            # paths_weighted_sum: [num_ent_pairs, full_encoder_dim]
            scores = self.fc(paths_weighted_sum)
            probs = self.sigmoid(scores).squeeze(dim=1)
        elif self.pooling_method == "sat":
            # h: [num_ent_pairs x num_paths, full_encoder_dim]
            path_hiddens = self.path_att(h)
            # path_hiddens: [num_ent_pairs x num_paths, path_hidden_dim]
            att = self.att(self.relu(path_hiddens + self.path_context))
            # att: [num_ent_pairs x num_paths, 1]
            att = att.view(num_ent_pairs, num_paths, -1)
            path_weights = self.softmax(att)
            paths_feats = h.view(num_ent_pairs, num_paths, -1)
            paths_weighted_sum = (paths_feats * path_weights).sum(dim=1)
            # paths_weighted_sum: [num_ent_pairs, full_encoder_dim]
            scores = self.fc(paths_weighted_sum)
            probs = self.sigmoid(scores).squeeze(dim=1)

        # visualization
        path_weights = path_weights.view(num_ent_pairs, num_paths)
        type_weights = alphas.view(num_ent_pairs, num_paths, num_steps, num_types)

        return probs, path_weights, type_weights

