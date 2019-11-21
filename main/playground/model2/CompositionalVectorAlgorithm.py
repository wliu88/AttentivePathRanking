import time
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import pickle
from tqdm import tqdm
import os
import json
from collections import OrderedDict, defaultdict
from scipy.stats import kurtosis, skew
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from main.playground.model2.CompositionalVectorSpaceModel import CompositionalVectorSpaceModel
from main.playground.BatcherFileList import BatcherFileList
from main.experiments.Metrics import compute_scores
from main.playground.Logger import Logger
from main.playground.Visualizer import Visualizer


class CompositionalVectorAlgorithm:

    def __init__(self, dataset, experiment_dir, entity_type2vec_filename, learning_rate=0.1, weight_decay=0.0001,
                 number_of_epochs=30, learning_rate_step_size=50, learning_rate_decay=0.5, visualize=False,
                 best_models=None, pooling_method="sat", attention_method="sat", early_stopping_metric="map",
                 mid2name_filename=None, calculate_path_attn_stats=False, calculate_type_attn_stats=False):
        """
        This class is used to run Attentive Path Ranking algorithm. The training progress is logged in tensorboardx.

        :param dataset:
        :param experiment_dir:
        :param entity_type2vec_filename:
        :param learning_rate:
        :param weight_decay:
        :param number_of_epochs:
        :param learning_rate_step_size:
        :param learning_rate_decay:
        :param visualize: if set to true, save visualized paths to folder
        :param best_models: if provided, models will only be trained to the epochs of the best models. This is mainly
                            used for visualizing paths after all models have been trained fully once.
        :param pooling_method: "sat", "lse", "avg", or "max"
        :param attention_method: "sat", "specific", or "abstract"
        :param early_stopping_metric: "map" or "accuracy"
        :param mid2name_filename:
        :param calculate_path_attn_stats:
        :param calculate_type_attn_stats:
        """
        self.dataset = dataset
        assert dataset == "wordnet" or dataset == "freebase"

        self.attention_method = attention_method
        self.pooling_method = pooling_method
        self.early_stopping_metric = early_stopping_metric

        self.entity_type2vec_filename = entity_type2vec_filename
        self.input_dirs = []
        self.entity_vocab = None
        self.relation_vocab = None
        self.entity_type_vocab = None
        self.experiment_dir = experiment_dir
        self.load_data(experiment_dir)

        self.logger = Logger()

        # for visualizing results
        self.best_models = best_models
        self.visualize = visualize
        self.calculate_path_attn_stats = calculate_path_attn_stats
        self.calculate_type_attn_stats = calculate_type_attn_stats

        if calculate_path_attn_stats:
            self.path_weights_dir = os.path.join(self.experiment_dir, "path_weights")
            if not os.path.exists(self.path_weights_dir):
                os.mkdir(self.path_weights_dir)

        if calculate_type_attn_stats:
            self.type_weights_dir = os.path.join(self.experiment_dir, "type_weights")
            if not os.path.exists(self.type_weights_dir):
                os.mkdir(self.type_weights_dir)

        self.idx2entity = {v: k for k, v in self.entity_vocab.items()}
        self.idx2entity_type = {v: k for k, v in self.entity_type_vocab.items()}
        self.idx2relation = {v: k for k, v in self.relation_vocab.items()}
        self.visualizer = Visualizer(self.idx2entity, self.idx2entity_type, self.idx2relation,
                                     save_dir=os.path.join(experiment_dir, "results"),
                                     mid2name_filename=mid2name_filename)

        self.all_best_epoch_val_test = {}
        # best_epoch_val_test = {"epoch": -1, "val_acc": -1, "val_ap": -1, "test_acc": -1, "test_ap": -1}
        self.number_of_epochs = number_of_epochs

    def load_data(self, experiment_dir):
        data_dir = os.path.join(experiment_dir, "data")
        for folder in os.listdir(data_dir):
            if "data_output" in folder:
                input_dir = os.path.join(data_dir, folder)
                for fld in os.listdir(input_dir):
                    self.input_dirs.append(os.path.join(input_dir, fld))
            if "vocab" in folder:
                vocab_dir = os.path.join(data_dir, folder)
                for fld in os.listdir(vocab_dir):
                    if "entity_type_vocab" in fld:
                        entity_type_vocab_filename = os.path.join(vocab_dir, fld)
                        entity_type_vocab = json.load(open(entity_type_vocab_filename, "r"))
                        self.entity_type_vocab = entity_type_vocab
                    if "entity_vocab" in fld:
                        entity_vocab_filename = os.path.join(vocab_dir, fld)
                        self.entity_vocab = json.load(open(entity_vocab_filename, "r"))
                    if "relation_vocab" in fld:
                        relation_vocab_filename = os.path.join(vocab_dir, fld)
                        self.relation_vocab = json.load(open(relation_vocab_filename, "r"))

    def train_and_test(self):
        print(self.input_dirs)
        for input_dir in self.input_dirs:
            self.train(input_dir)

        # print statistics
        print(self.all_best_epoch_val_test)
        accs = []
        aps = []
        for rel in self.all_best_epoch_val_test:
            best_model_score = self.all_best_epoch_val_test[rel]
            accs.append(best_model_score["test_acc"])
            aps.append(best_model_score["test_ap"])
        print("Average Accuracy:", sum(accs)/len(accs))
        print("Mean Average Precision:", sum(aps) / len(aps))

    def train(self, input_dir):
        print("Setting up model")
        # default parameters: relation_embedding_dim=50, entity_embedding_dim=0, entity_type_embedding_dim=300,
        #                     attention_dim = 50, relation_encoder_dim=150, full_encoder_dim=150

        if self.dataset == "wordnet":
            entity_type_embedding_dim = 300
        else:
            entity_type_embedding_dim = 50
        model = CompositionalVectorSpaceModel(relation_vocab_size=len(self.relation_vocab),
                                              entity_vocab_size=len(self.entity_vocab),
                                              entity_type_vocab_size=len(self.entity_type_vocab),
                                              relation_embedding_dim=50,
                                              entity_embedding_dim=0,
                                              entity_type_embedding_dim=entity_type_embedding_dim,
                                              entity_type_vocab=self.entity_type_vocab,
                                              entity_type2vec_filename=self.entity_type2vec_filename,
                                              attention_dim=50,
                                              relation_encoder_dim=150,
                                              full_encoder_dim=150,
                                              pooling_method=self.pooling_method,
                                              attention_method=self.attention_method)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=learning_rate_step_size, gamma=learning_rate_decay)
        optimizer = optim.Adam(model.parameters())
        criterion = torch.nn.BCELoss().cuda()

        best_epoch_val_test = {"epoch": -1, "val_acc": -1, "val_ap": -1, "test_acc": -1, "test_ap": -1}
        rel = input_dir.split("/")[-1]
        train_files_dir = os.path.join(input_dir, "train")
        val_files_dir = os.path.join(input_dir, "dev")
        test_files_dir = os.path.join(input_dir, "test")
        print("Setting up train, validation, and test batcher...")
        train_batcher = BatcherFileList(train_files_dir, batch_size=16, shuffle=True, max_number_batchers_on_gpu=100)
        val_batcher = BatcherFileList(val_files_dir, batch_size=16, shuffle=False, max_number_batchers_on_gpu=100)
        test_batcher = BatcherFileList(test_files_dir, batch_size=16, shuffle=True, max_number_batchers_on_gpu=100)

        count = 0
        while True:
            data = train_batcher.get_batch()
            if data is None:
                break
            count += 1

        run_epochs = 0
        if self.best_models is not None:
            run_epochs = self.best_models[rel]["epoch"] + 1
        else:
            run_epochs = self.number_of_epochs

        # 1. training process
        for epoch in range(run_epochs):
            # self.scheduler.step()
            total_loss = 0
            start = time.time()

            # for i in tqdm(range(count + 1)):
            for i in range(count + 1):
                data = train_batcher.get_batch()
                if data is not None:

                    inputs, labels = data
                    model.train()
                    model.zero_grad()
                    probs, path_weights, type_weights = model(inputs)
                    loss = criterion(probs, labels)

                    loss.backward()
                    # IMPORTANT: grad clipping is important if loss is large. May not be necessary for LSTM
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    total_loss += loss.item()

            time.sleep(1)
            print("Epoch", epoch, "spent", time.time() - start, "with total loss:", total_loss)

            # compute scores, record best scores, and generate visualizations on the go
            if self.best_models is None:
                # compute train, validation, and test scores and log in tensorboardx
                train_acc, train_ap = self.score_and_visualize(model, train_batcher, rel, "train", epoch)
                val_acc, val_ap = self.score_and_visualize(model, val_batcher, rel, "val", epoch)
                test_acc, test_ap = self.score_and_visualize(model, test_batcher, rel, "test", epoch)
                # log training progress on tensorboardx
                self.logger.log_loss(total_loss, epoch, rel)
                self.logger.log_accuracy(train_acc, val_acc, test_acc, epoch, rel)
                self.logger.log_ap(train_ap, val_ap, test_ap, epoch, rel)
                for name, param in model.named_parameters():
                    self.logger.log_param(name, param, epoch)

                # selecting the best model based on performance on validation set
                if self.early_stopping_metric == "accuracy":
                    if val_acc > best_epoch_val_test["val_acc"]:
                        best_epoch_val_test = {"epoch": epoch,
                                               "val_acc": val_acc, "val_ap": val_ap,
                                               "test_acc": test_acc, "test_ap": test_ap}
                elif self.early_stopping_metric == "map":
                    if val_ap > best_epoch_val_test["val_ap"]:
                        best_epoch_val_test = {"epoch": epoch,
                                               "val_acc": val_acc, "val_ap": val_ap,
                                               "test_acc": test_acc, "test_ap": test_ap}
                else:
                    raise Exception("Early stopping metric not recognized.")

                # Stop training if loss has reduced to zero
                if total_loss == 0:
                    break

            else:
                # only compute train and test scores for the best models
                if epoch == self.best_models[rel]["epoch"]:
                    train_acc, train_ap = self.score_and_visualize(model, train_batcher, rel, "train", epoch)
                    test_acc, test_ap = self.score_and_visualize(model, test_batcher, rel, "test", epoch)

        # 2. save best model
        if self.best_models is None:
            print("Best model", best_epoch_val_test)
            if self.visualize:
                self.visualizer.save_space(rel, best_epoch_val_test["epoch"])
            self.all_best_epoch_val_test[rel] = best_epoch_val_test

    def test(self, input_dir):
        test_files_dir = os.path.join(input_dir, "test")
        print("Setting up test batcher")
        batcher = BatcherFileList(test_files_dir, batch_size=16, shuffle=True, max_number_batchers_on_gpu=100)

        acc, ap = self.score_and_visualize(batcher)
        print("Total accuracy for testing set:", acc)
        print("AP for this relation:", ap)

    def score_and_visualize(self, model, batcher, rel, split, epoch):
        # store groundtruths and predictions
        score_instances = []
        # store various path stats for all entity pairs
        path_weights_stats = defaultdict(list)
        all_path_weights = None
        all_type_weights = None
        type_weights_sum = None
        type_weights_count = 0

        with torch.no_grad():
            model.eval()
            batcher.reset()
            while True:
                data = batcher.get_batch()
                if data is None:
                    break
                inputs, labels = data
                probs, path_weights, type_weights = model(inputs)

                if self.visualize and split == "test":
                    if (self.best_models is None) or (epoch == self.best_models[rel]["epoch"]):
                        # Visualizations
                        #   (1) show top k paths with highest weighted types.
                        #   (2) show only one path with detailed attention to each type in type hierarchies.
                        #   (3) show examples with same relation paths but different proposed path patterns.

                        # self.visualizer.visualize_paths_with_relation_and_type(inputs.clone().cpu().data.numpy(),
                        #                                                        labels.clone().cpu().data.numpy(),
                        #                                                        type_weights.clone().cpu().data.numpy(),
                        #                                                        path_weights.clone().cpu().data.numpy(),
                        #                                                        rel, split, epoch,
                        #                                                        filter_negative_example=True,
                        #                                                        filter_false_prediction=True,
                        #                                                        probs=probs.clone().cpu().data.numpy(),
                        #                                                        top_k_path=5,
                        #                                                        minimal_path_weight=0.2)
                        # self.visualizer.visualize_paths(inputs.clone().cpu().data.numpy(),
                        #                                 labels.clone().cpu().data.numpy(),
                        #                                 type_weights.clone().cpu().data.numpy(),
                        #                                 path_weights.clone().cpu().data.numpy(),
                        #                                 rel, split, epoch,
                        #                                 filter_negative_example=True,
                        #                                 filter_false_prediction=True,
                        #                                 probs=probs.clone().cpu().data.numpy(),
                        #                                 top_k_path=5,
                        #                                 minimal_path_weight=0.2)

                        self.visualizer.visualize_contradictions(inputs.clone().cpu().data.numpy(),
                                                                 labels.clone().cpu().data.numpy(),
                                                                 type_weights.clone().cpu().data.numpy(),
                                                                 path_weights.clone().cpu().data.numpy(),
                                                                 rel, split,
                                                                 filter_false_prediction=True,
                                                                 probs=probs.clone().cpu().data.numpy(),
                                                                 minimal_path_weight=0.15)

                # Visualize attention stats
                if self.calculate_type_attn_stats and split == "test":
                    # type_weights: [num_ent_pairs, num_paths, num_steps, num_types]
                    num_ent_pairs, num_paths, num_steps, num_types = type_weights.shape
                    if type_weights_sum is None:
                        type_weights_sum = torch.sum(type_weights.view(-1, num_types), dim=0)
                    else:
                        type_weights_sum += torch.sum(type_weights.view(-1, num_types), dim=0)
                    type_weights_count += num_ent_pairs * num_paths * num_steps

                    # # store all type weights
                    # type_weights = type_weights.view(-1, num_types).clone().cpu().data.numpy()
                    # if all_type_weights is None:
                    #     all_type_weights = type_weights
                    # else:
                    #     all_type_weights = np.vstack([all_type_weights, type_weights])

                if self.calculate_path_attn_stats and split == "test":
                    path_weights = path_weights.clone().cpu().data.numpy()
                    num_ent_pairs, num_paths = path_weights.shape

                    # normalize path weights for plotting
                    if num_paths > 1:
                        path_weights_sorted = np.sort(path_weights, axis=1)
                        path_weights_sorted = path_weights_sorted / np.max(path_weights_sorted, axis=1).reshape(num_ent_pairs, 1)
                        x_old = np.array(range(num_paths))
                        x_new = np.linspace(0, num_paths-1, 200)
                        func = interp1d(x_old, path_weights_sorted, axis=1)
                        path_weights_normalized = func(x_new)
                        if all_path_weights is None:
                            all_path_weights = path_weights_normalized
                        else:
                            all_path_weights = np.vstack([all_path_weights, path_weights_normalized])

                    # basic stats
                    # all_path_weights: [num_ent_pairs, num_paths]
                    # path_weights_stats["min"].extend(np.nanmin(all_path_weights, axis=1))
                    # path_weights_stats["max"].extend(np.nanmax(all_path_weights, axis=1))
                    # path_weights_stats["mean"].extend(np.nanmean(all_path_weights, axis=1))
                    # path_weights_stats["std"].extend(np.nanstd(all_path_weights, axis=1))
                    #
                    # #
                    # num_ent_pairs, num_paths = all_path_weights.shape
                    # for percent in [25, 50, 75]:
                    #     percentile = np.nanpercentile(all_path_weights, percent, axis=1).reshape(num_ent_pairs, -1)
                    #     smaller_paths_percentile = all_path_weights * (all_path_weights < percentile)
                    #     sum_paths_percentile = np.sum(smaller_paths_percentile, axis=1)
                    #     path_weights_stats["paths_" + str(percent)].extend(sum_paths_percentile)

                    # measure of tails
                    # path_weights_stats["skew"].extend(skew(all_path_weights, axis=1))
                    # path_weights_stats["kurtosis"].extend(kurtosis(all_path_weights, axis=1))

                for label, prob in zip(labels, probs):
                    score_instances.append((None, label.item(), prob.item()))
                # print("accuracy for this batch of", inputs.shape[0], "examples is", num_correct / inputs.shape[0])
                # print("Total accuracy for training set:", total_num_correct / total_pairs)

        # summarize scores and stats
        ap, rr, acc = compute_scores(score_instances)
        # print("AP for this relation:", ap)

        if self.visualize and split == "test":
            self.visualizer.print_contradictions(rel)

        if self.calculate_type_attn_stats and split == "test":
            if type_weights_sum is not None:
                print("Average type attention weights for {} {}".format(rel, split),
                      type_weights_sum / type_weights_count)

            if all_type_weights is not None:
                pass
                # # save type weights to file
                # type_weights_file = os.path.join(self.type_weights_dir, "{}_{}.csv".format(rel, split))
                # np.savetxt(type_weights_file, all_type_weights, delimiter=",", fmt='%.6e')

        if self.calculate_path_attn_stats and split == "test":
            path_stats = OrderedDict()
            # all_path_weights[all_path_weights == 0] = float("nan")
            # path_stats["min"] = np.average(np.array(path_weights_stats["min"]))
            # path_stats["max"] = np.average(np.array(path_weights_stats["max"]))
            # path_stats["mean_mean"] = np.mean(np.array(path_weights_stats["mean"]))
            # path_stats["mean_std"] = np.std(np.array(path_weights_stats["mean"]))
            # path_stats["std_mean"] = np.mean(np.array(path_weights_stats["std"]))
            # path_stats["std_std"] = np.std(np.array(path_weights_stats["std"]))
            #
            # for percent in [25, 50, 75]:
            #     path_stats["paths_" + str(percent) + "_mean"] = np.mean(np.array(path_weights_stats["paths_" + str(percent)]))
            #     path_stats["paths_" + str(percent) + "_std"] = np.std(np.array(path_weights_stats["paths_" + str(percent)]))

            # path_stats["skew_mean"] = np.average(np.array(path_weights_stats["skew"]))
            # path_stats["skew_std"] = np.std(np.array(path_weights_stats["skew"]))
            # path_stats["kurtosis_mean"] = np.average(np.array(path_weights_stats["kurtosis"]))
            # path_stats["kurtosis_std"] = np.std(np.array(path_weights_stats["kurtosis"]))
            #
            # print("Path weights stats:", path_stats)

            # plot path weights
            if all_path_weights is not None:
                # visualize path weights
                path_visualization_file = os.path.join(self.path_weights_dir, "{}_{}.png".format(rel, split))
                path_weights_total_avg = np.mean(all_path_weights, axis=0)
                print(path_weights_total_avg)
                plt.plot(range(200), path_weights_total_avg)
                plt.savefig(path_visualization_file)
                plt.cla()
                plt.close()

                # save path weights to file
                path_weights_file = os.path.join(self.path_weights_dir, "{}_{}.csv".format(rel, split))
                np.savetxt(path_weights_file, all_path_weights, delimiter=",", fmt='%.6e')

        return acc, ap