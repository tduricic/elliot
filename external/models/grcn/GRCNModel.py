"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from abc import ABC

from .GraphRefiningLayer import GraphRefiningLayer
from .GraphConvolutionalLayer import GraphConvolutionalLayer
import torch
import torch_geometric
import numpy as np
import random


class GRCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 embed_k_multimod,
                 l_w,
                 num_layers,
                 num_routings,
                 modalities,
                 aggregation,
                 weight_mode,
                 multimodal_features,
                 adj,
                 adj_user,
                 rows,
                 cols,
                 size_rows,
                 random_seed,
                 name="GRCN",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.embed_k_multimod = embed_k_multimod
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.modalities = modalities
        self.aggregation = aggregation
        self.weight_mode = weight_mode
        self.n_layers = num_layers
        self.n_routings = num_routings
        self.adj = adj
        self.adj_user = adj_user
        self.rows = torch.tensor(rows, dtype=torch.int64)
        self.cols = torch.tensor(cols, dtype=torch.int64)
        self.size_rows = torch.tensor(size_rows, dtype=torch.int64)

        # collaborative embeddings
        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        # multimodal collaborative embeddings
        self.Gum = dict()
        self.Gim = dict()
        self.multimodal_features_shapes = [mf.shape[1] for mf in multimodal_features]
        for m_id, m in enumerate(modalities):
            self.Gum[m] = torch.nn.Parameter(
                torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k_multimod)))
            )
            self.Gum[m].to(self.device)
            self.Gim[m] = torch.nn.functional.normalize(torch.tensor(multimodal_features[m_id], dtype=torch.float32))
            self.Gim[m].to(self.device)

        propagation_graph_convolutional_network_list = []
        for layer in range(self.n_layers):
            propagation_graph_convolutional_network_list.append((
                GraphConvolutionalLayer(), 'x, edge_index -> x'
            ))
        self.propagation_graph_convolutional_network = torch_geometric.nn.Sequential(
            'x, edge_index',
            propagation_graph_convolutional_network_list
        )
        self.propagation_graph_convolutional_network.to(self.device)

        self.projection_multimodal = dict()
        self.propagation_graph_refining_network = dict()

        for m_id, m in enumerate(self.modalities):
            self.projection_multimodal[m] = torch.nn.Linear(in_features=self.multimodal_features_shapes[m_id],
                                                            out_features=self.embed_k_multimod)
            propagation_graph_refining_network_list = []
            for layer in range(self.n_routings + 1):
                propagation_graph_refining_network_list.append(
                    (GraphRefiningLayer(self.rows, self.size_rows), 'x, edge_index -> x'))

            self.propagation_graph_refining_network[m] = torch_geometric.nn.Sequential(
                'x, edge_index', propagation_graph_refining_network_list)
            self.propagation_graph_refining_network[m].to(self.device)

        self.softplus = torch.nn.Softplus()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.lr_scheduler = self.set_lr_scheduler()

    def set_lr_scheduler(self):
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.96 ** (epoch / 50))
        return scheduler

    def propagate_embeddings(self):
        # Graph Refining Layers
        x_all_m = dict()
        alphas_m = dict()
        gum = torch.empty((len(self.modalities), self.num_users, self.embed_k_multimod))
        gim = torch.empty((len(self.modalities), self.num_items, self.embed_k_multimod))
        for m_id, m in enumerate(self.modalities):
            gum[m_id] = torch.nn.functional.normalize(self.Gum[m].to(self.device))
            gim[m_id] = torch.nn.functional.normalize(
                torch.nn.functional.leaky_relu(self.projection_multimodal[m](self.Gim[m])).to(self.device))
            cols_attr = gim[m_id][self.cols - self.num_users]

            for t in range(self.n_routings):
                rows_attr = gum[m_id][self.rows]
                x_all_m[m] = torch.cat((gum[m_id], gim[m_id]), dim=0)
                x_all_hat_m = list(
                    self.propagation_graph_refining_network[m].children()
                )[t](x_all_m[m].to(self.device), rows_attr.to(self.device), cols_attr.to(self.device),
                     self.adj_user.to(self.device))
                gum[m_id] += x_all_hat_m[:self.num_users]
                gum[m_id] = torch.nn.functional.normalize(gum[m_id])

            rows_attr = torch.cat((gum[m_id][self.rows], gim[m_id][self.cols - self.num_users]), dim=0)
            cols_attr = torch.cat((gim[m_id][self.cols - self.num_users], gum[m_id][self.rows]), dim=0)
            x_all_m[m] = torch.cat((gum[m_id], gim[m_id]), dim=0)
            x_all_hat_m = list(
                self.propagation_graph_refining_network[m].children()
            )[self.n_routings](x_all_m[m].to(self.device), rows_attr.to(self.device), cols_attr.to(self.device),
                               self.adj.to(self.device))
            x_all_m[m] += x_all_hat_m
            alphas_m[m] = list(self.propagation_graph_refining_network[m].children())[-1].alpha

        # Graph Convolutional Layers
        x_all = torch.cat([value for _, value in x_all_m.items()], dim=1)
        if self.weight_mode == 'mean':
            alphas = torch.cat([torch.unsqueeze(value, dim=0) for _, value in alphas_m.items()], dim=0)
            alphas = torch.sum(alphas, dim=0) / len(self.modalities)
        elif self.weight_mode == 'max':
            alphas = torch.cat([torch.unsqueeze(value, dim=0) for _, value in alphas_m.items()], dim=0)
            alphas, _ = torch.max(alphas, dim=0)
        else:
            raise NotImplementedError('This weight mode has not been implemented yet!')

        alphas = torch.relu(alphas)

        ego_embeddings = torch.cat((self.Gu.to(self.device), self.Gi.to(self.device)), 0)
        all_embeddings = [torch.nn.functional.normalize(ego_embeddings)]
        for layer in range(self.n_layers):
            all_embeddings += [torch.nn.functional.normalize(
                list(self.propagation_graph_convolutional_network.children())[layer](
                    all_embeddings[layer].to(self.device),
                    self.adj.to(self.device),
                    alphas.to(self.device)))]
        all_embeddings = torch.cat([torch.unsqueeze(value, dim=0) for value in all_embeddings])
        all_embeddings = torch.sum(all_embeddings, dim=0)
        x = torch.cat([all_embeddings, x_all], dim=1)
        gu, gi = torch.split(x, [self.num_users, self.num_items], 0)
        return gu, gi

    def forward(self, inputs, **kwargs):
        gu, gi = inputs
        gamma_u = torch.squeeze(gu).to(self.device)
        gamma_i = torch.squeeze(gi).to(self.device)

        xui = torch.sum(gamma_u * gamma_i, 1)

        return xui, gamma_u, gamma_i

    def predict(self, gu, gi, **kwargs):
        return torch.matmul(gu.to(self.device), torch.transpose(gi.to(self.device), 0, 1))

    def train_step(self, batch):
        gu, gi = self.propagate_embeddings()
        user, pos, neg = batch
        xu_pos, gamma_u, gamma_i_pos = self.forward(inputs=(gu[user], gi[pos]))
        xu_neg, _, gamma_i_neg = self.forward(inputs=(gu[user], gi[neg]))

        difference = torch.clamp(xu_pos - xu_neg, -80.0, 1e8)
        loss = torch.mean(torch.nn.functional.softplus(-difference))
        reg_loss = self.l_w * (1 / 2) * (gamma_u.norm(2).pow(2) +
                                         gamma_i_pos.norm(2).pow(2) +
                                         gamma_i_neg.norm(2).pow(2)) / user.shape[0]
        loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)