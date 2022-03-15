import typing as t
import os
import numpy as np
import torch
import json
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class SentimentInteractionsTextualAttributes(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.interactions_sim = getattr(ns, "interactions_sim", None)

        self.item_mapping = {}
        self.user_mapping = {}

        inner_items = self.check_interactions_in_folder()

        self.users = users
        self.items = items & inner_items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "SentimentInteractionsTextualAttributes"
        ns.object = self
        ns.textual_interactions_sim = self.interactions_sim

        ns.user_mapping = self.user_mapping
        ns.item_mapping = self.item_mapping

        return ns

    def check_interactions_in_folder(self) -> (t.Set[int]):
        items = set()
        if self.interactions_sim:
            with open(self.interactions_sim, 'r') as f:
                int_sim = json.load(f)
            items_from_json = [[int(k.split('_')[0]), int(k.split('_')[1])] for k in int_sim]
            items_from_json = set([item for sublist in items_from_json for item in sublist])
            items = items.union(items_from_json)

        if items:
            self.item_mapping = {item: val for val, item in enumerate(items)}

        return items

    def get_all_features(self, public_items):
        def add_zero(s):
            zeros_to_add = num_digits - len(s)
            return ''.join(['0' for _ in range(zeros_to_add)]) + s
        with open(self.interactions_sim, 'r') as f:
            int_sim = json.load(f)
        num_digits = len(str(int(list(public_items.keys())[-1])))
        all_interactions_private = []
        all_interactions_public = ['_'.join(
            [add_zero(str(public_items[float(inter[:-4].split('_')[0])])),
             add_zero(str(public_items[float(inter[:-4].split('_')[1])]))]) for inter in
            file_list]
        sorted_indices_public = sorted(range(len(all_interactions_public)), key=lambda k: all_interactions_public[k])
        all_interactions_private = [all_interactions_private[index] for index in sorted_indices_public]
        return all_interactions_private
