import operator

import numpy as np

from elliot.evaluation.evaluator import Evaluator
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.folder import build_model_folder
from elliot.utils.write import store_recommendation
from elliot.recommender.base_recommender_model import init_charger

from .GraphRecModel import GraphRecModel


import torch.utils.data
import torch
import random
import os

np.random.seed(0)

class GraphRec(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Graph neural networks for social recommendation

        GraphRec presented by Fan et al. in `paper <https://dl.acm.org/doi/pdf/10.1145/3308558.3313488>`

        Args:
            meta:
                verbose: Field to enable verbose logs
                save_recs: Field to enable recommendation lists storage
                validation_metric: Metric used for model validation
            epochs: Number of epochs
            batch_size: Batch sizer
            lr: Learning rate
            factors: Number of latent factors

        To include the recommendation model, add it to the config file adopting the following pattern:

        .. code:: yaml

        models:
            external.GraphRec:
            meta:
                verbose: True
                save_recs: False
                validation_metric: nDCG@10
            epochs: 10
            batch_size: 128
            lr: 0.001
            factors: 64
        """
        self._params_list = [
            ("_epochs", "epochs", "epochs", 10, int, None),
            ("_batch_size", "batch_size", "batch_size", 128, int, None),
            ("_test_batch_size", "test_batch_size", "test_batch_size", 1000, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_sc_filepath", "social_connections_filepath", "social_connections_filepath", None, None, None)
        ]
        self.autoset_params()

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self._use_cuda = False
        if torch.cuda.is_available():
            self._use_cuda = True
        self._device = torch.device("cuda" if self._use_cuda else "cpu")

        # TODO what is still missing is to make a different implementation based on if there is or there isn't a validation set
        self._history_u_lists, self._history_ur_lists, self._history_v_lists, self._history_vr_lists, \
        self._train_u, self._train_v, self._train_r, self._val_u, self._val_v, self._val_r, \
        self._test_u, self._test_v, self._test_r, self._social_adj_lists, self._ratings_list = \
            self.preprocess_data(data.train_dict, data.val_dict, data.test_dict, params.social_connections_filepath)


        self._embed_dim = params.factors

        self._trainset = torch.utils.data.TensorDataset(torch.LongTensor(self._train_u), torch.LongTensor(self._train_v),
                                                  torch.FloatTensor(self._train_r))
        # self._valset = torch.utils.data.TensorDataset(torch.LongTensor(self._val_u),
        #                                                torch.LongTensor(self._val_v),
        #                                                torch.FloatTensor(self._val_r))
        # self._testset = torch.utils.data.TensorDataset(torch.LongTensor(self._test_u), torch.LongTensor(self._test_v),
        #                                         torch.FloatTensor(self._test_r))

        self._train_loader = torch.utils.data.DataLoader(self._trainset, batch_size=params.batch_size, shuffle=True)
        # self._val_loader = torch.utils.data.DataLoader(self._valset, batch_size=params.batch_size, shuffle=True)
        # self._test_loader = torch.utils.data.DataLoader(self._testset, batch_size=params.batch_size, shuffle=True)
        # self._num_users = self._history_u_lists.__len__()
        # self._num_items = self._history_v_lists.__len__()
        # self._num_users = max(list(self._history_u_lists.keys()))+1
        # self._num_items = max(list(self._history_v_lists.keys()))+1
        # self._num_users = len(set(self._train_u + self._val_u + self._test_u))
        # self._num_items = len(set(self._train_v + self._val_v + self._test_v))
        self._num_users = max(set(self._train_u + self._val_u + self._test_u))+1
        self._num_items = max(set(self._train_v + self._val_v + self._test_v))+1

        #self._unique_users = set(self._train_u + self._val_u + self._test_u)

        self._num_ratings = self._ratings_list.__len__()

        self._model = GraphRecModel(num_users=self._num_users,
                               num_items=self._num_items,
                               num_ratings=self._num_ratings,
                               learning_rate=params.lr,
                               embed_dim=self._embed_dim,
                               use_cuda=self._use_cuda,
                               history_u_lists=self._history_u_lists,
                               history_ur_lists=self._history_ur_lists,
                               history_v_lists=self._history_v_lists,
                               history_vr_lists=self._history_vr_lists,
                               social_adj_lists=self._social_adj_lists,
                               random_seed=self._seed).to(self._device)
        #self._model = GraphRec(self._enc_u, self._enc_v_history, self._r2e, random_seed=self._seed).to(self._device)

    @property
    def name(self):
        return "GraphRec" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            print(it)
            self._model.train()
            running_loss = 0.0
            for i, train_data in enumerate(self._train_loader, 0):
                batch_nodes_u, batch_nodes_v, labels_list = train_data
                self._model.optimizer.zero_grad()
                loss = self._model.loss(batch_nodes_u.to(self._device), batch_nodes_v.to(self._device), labels_list.to(self._device))
                loss.backward(retain_graph=True)
                self._model.optimizer.step()
                running_loss += loss.item()
            if it % 5 == 0:
                self.evaluate(it, running_loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        with torch.no_grad():
            v = torch.tensor(self._data.items).to(self._device)
            for index, offset in enumerate(range(0, len(self._data.users), self._batch_size)):
                offset_stop = min(offset + self._batch_size, len(self._data.users))
                predictions = None
                for user_index in range(offset, offset_stop):
                    user_id = self._data.users[user_index]
                    # u_predictions = np.repeat(-np.inf, len(self._data.items))
                    # if user_id not in self._history_u_lists:
                        #if predictions is None:
                        #    predictions = np.array(u_predictions)
                        #else:
                        #    predictions = np.vstack((predictions, u_predictions))
                    #    continue
                    u = torch.tensor(np.repeat(user_id, len(self._data.items))).to(self._device)

                    u_predictions = self._model.forward(u, v).data.cpu().numpy()
                    if predictions is None:
                        predictions = np.array(u_predictions)
                    else:
                        predictions = np.vstack((predictions, u_predictions))
                recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
                predictions_top_k_val.update(recs_val)
                predictions_top_k_test.update(recs_test)
        self._model.train()
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], self._device, k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath (
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")


    def preprocess_data(self, train_dict, val_dict, test_dict, social_connections_filepath):
        history_u_lists = self.create_history_u_lists(train_dict)
        history_ur_lists = self.create_history_ur_lists(train_dict)
        history_v_lists = self.create_history_v_lists(train_dict)
        history_vr_lists = self.create_history_vr_lists(train_dict)
        train_u, train_v, train_r = self.create_uvr(train_dict)
        val_u, val_v, val_r = self.create_uvr(val_dict)
        test_u, test_v, test_r  = self.create_uvr(test_dict)
        social_adj_lists = self.create_social_adj_lists(social_connections_filepath)
        ratings_list = self.create_ratings_list(train_r + val_r + test_r)

        return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
        train_u, train_v, train_r, val_u, val_v, val_r, test_u, test_v, test_r, \
        social_adj_lists, ratings_list


    def create_history_u_lists(self, user_item_ratings_dict):
        history_u_lists = {}
        for user_id in user_item_ratings_dict:
            history_u_lists[user_id] = list(user_item_ratings_dict[user_id].keys())
        return history_u_lists

    def create_history_ur_lists(self, user_item_ratings_dict):
        history_ur_lists = {}
        for user_id in user_item_ratings_dict:
            history_ur_lists[user_id] = list(user_item_ratings_dict[user_id].values())
        return history_ur_lists

    def create_history_v_lists(self, user_item_ratings_dict):
        history_v_lists = {}
        for user_id in user_item_ratings_dict:
            for item_id in user_item_ratings_dict[user_id]:
                if item_id not in history_v_lists:
                    history_v_lists[item_id] = []
                history_v_lists[item_id].append(user_id)
        return history_v_lists

    def create_history_vr_lists(self, user_item_ratings_dict):
        history_vr_lists = {}
        for user_id in user_item_ratings_dict:
            for item_id in user_item_ratings_dict[user_id]:
                if item_id not in history_vr_lists:
                    history_vr_lists[item_id] = []
                history_vr_lists[item_id].append(user_item_ratings_dict[user_id][item_id])
        return history_vr_lists

    def create_uvr(self, user_item_ratings_dict):
        uvr_list = []
        u_list = []
        v_list = []
        r_list = []
        for user_id in user_item_ratings_dict:
            for item_id in user_item_ratings_dict[user_id]:
                uvr_list.append((user_id, item_id, user_item_ratings_dict[user_id][item_id]))
        random.shuffle(uvr_list)
        for (u, v, r) in uvr_list:
            u_list.append(u)
            v_list.append(v)
            r_list.append(r)
        return u_list, v_list, r_list

    def create_social_adj_lists(self, social_connections_filepath):
        social_adj_lists = {}
        with open(social_connections_filepath) as f:
            for line in f:
                tokens = line.split('\t')
                user_1 = int(tokens[0])
                user_2 = int(tokens[1])
                weight = float(tokens[2])

                if user_1 not in social_adj_lists:
                    social_adj_lists[user_1] = set()
                if user_2 not in social_adj_lists:
                    social_adj_lists[user_2] = set()

                social_adj_lists[user_1].add(user_2)
                social_adj_lists[user_2].add(user_1)

        return social_adj_lists


    def create_ratings_list(self, ratings_list):
        return sorted(list(set(ratings_list)))