import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
import threading
import tensorflow as tf
from collections import Counter


from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin

from elliot.dataset.samplers import custom_sampler as cs

from .UserFeatureMapper import UserFeatureMapper
from .KGFlexModel import KGFlexModel
from .tfidf_utils import TFIDF

#mp.set_start_method('fork')


def uif_worker(us_f, its_f, mapping):
    uif = []
    lengths = []
    for it_f in its_f:
        s = set.intersection(set(map(lambda x: mapping[x], us_f)), it_f)
        lengths.append(len(s))
        uif.extend(list(s))
    return tf.RaggedTensor.from_row_lengths(uif, lengths)


class KGFlexUmap(RecMixin, BaseRecommenderModel):

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        # auto parameters
        self._params_list = [
            ("_lr", "lr", "lr", 0.01, None, None),
            ("_embedding", "embedding", "em", 10, int, None),
            ("_first_order_limit", "first_order_limit", "fol", -1, None, None),
            ("_second_order_limit", "second_order_limit", "sol", -1, None, None),
            ("_l_w", "l_w", "l_w", 0.1, float, None),
            ("_l_b", "l_b", "l_b", 0.001, float, None),
            ("_loader", "loader", "load", "KGFlex", None, None),
            ("_npr", "npr", "npr", 1, int, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._side = getattr(self._data.side_information, self._loader, None)
        first_order_limit = self._params.first_order_limit
        second_order_limit = self._params.second_order_limit

        self.logger.info('Preparing KGFlex experiment...')

        # ------------------------------ ITEM FEATURES ------------------------------
        self.logger.info('importing items features')

        uri_to_private = {self._side.mapping[i]: p for i, p in self._data.public_items.items()}

        item_features_df1 = pd.DataFrame()
        item_features_df1['item'] = self._side.triples['uri'].map(uri_to_private)
        item_features_df1['f'] = list(zip(self._side.triples['predicate'], self._side.triples['object']))
        item_features_df1 = item_features_df1.dropna()
        self.item_features1 = item_features_df1.groupby('item')['f'].apply(set).to_dict()

        item_features_df2 = pd.DataFrame()
        item_features_df2['item'] = self._side.second_order_features['uri_x'].map(uri_to_private)
        item_features_df2['f'] = list(
            zip(self._side.second_order_features['predicate_x'], self._side.second_order_features['predicate_y'],
                self._side.second_order_features['object_y']))
        item_features_df2 = item_features_df2.dropna()
        self.item_features2 = item_features_df2.groupby('item')['f'].apply(set).to_dict()

        self.item_features = pd.concat([item_features_df1, item_features_df2]).groupby('item')['f'].apply(set).to_dict()

        # ------------------------------ USER FEATURES ------------------------------
        # candidate_items = np.where(np.bincount(self._data.sp_i_train.indices) > self._data.num_users * 0.01)[0]

        self.user_feature_mapper = UserFeatureMapper(data=self._data,
                                                     item_features=self.item_features1,
                                                     item_features2=self.item_features2,
                                                     first_order_limit=first_order_limit,
                                                     second_order_limit=second_order_limit,
                                                     negative_positive_ratio=self._npr)

        # ------------------------------ MODEL FEATURES ------------------------------
        self.logger.info('Features mapping started')
        users_features = self.user_feature_mapper.users_features

        features = set()
        for _, f in users_features.items():
            features = set.union(features, set(f))

        self.num_features = len(features)
        feature_key_mapping = dict(zip(list(features), range(self.num_features)))

        self.logger.info('FEATURES INFO: {} features found'.format(len(feature_key_mapping)))

        # -------------------------- BUILDING CONTENT VECTORS --------------------------
        content_item_idxs = list()
        content_feature_idxs = list()
        for i in range(self._data.num_items):
            ifs = self.item_features[i]
            common = set.intersection(set(feature_key_mapping.keys()), ifs)
            for f in common:
                content_item_idxs.append(i)
                content_feature_idxs.append(feature_key_mapping[f])

        content_vectors = csr_matrix((np.ones(len(content_item_idxs)) + 1, (content_item_idxs, content_feature_idxs)),
                                     (self._data.num_items, self.num_features))

        # Plus one above is for using linear operator that subtracts 1

        # content_vectors = tf.sparse.SparseTensor(item_features, np.ones(len(item_features), dtype=np.float32),
        #                                          (self._data.num_items, self.num_features))

        # -------------------------- BUILDING USER-FEATURE VECTORS -------------------------

        user_feature_weights = tf.constant(
            [[users_features[u].get(f, 0) for f in features] for u in self._data.private_users])

        # def stack_ragged(tensors):
        #     values = tf.concat(tensors, axis=0)
        #     lens = tf.stack([tf.shape(t, out_type=tf.int64)[0] for t in tensors])
        #     return tf.RaggedTensor.from_row_lengths(values, lens)
        #
        # item_features = stack_ragged(item_features)

        self.logger.info('Starting KGFlex experiment...')

        self._sampler = cs.Sampler(self._data.i_train_dict)

        # ------------------------------ MODEL ------------------------------
        self._model = KGFlexModel(num_users=self._data.num_users,
                                  num_items=self._data.num_items,
                                  user_feature_weights=user_feature_weights,
                                  content_vectors=content_vectors,
                                  num_features=self.num_features,
                                  factors=self._embedding,
                                  learning_rate=self._lr)

    @property
    def name(self):
        return "KGFlexUmap" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def get_recommendations(self, k: int = 10):
        predictions = self._model.get_all_recs()
        return self._model.get_all_topks(predictions, self.get_candidate_mask(validation=True), k,
                                         self._data.private_users, self._data.private_items) if hasattr(self._data,
                                                                                                        "val_dict") else {}, self._model.get_all_topks(
            predictions, self.get_candidate_mask(), k, self._data.private_users, self._data.private_items)

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()
            self.evaluate(it, loss)