import copy
import pathlib
import pickle

import dgl
import networkx as nx
import numpy as np
import torch
import torch.utils.data

from . import tour_cost, fixed_edge_tour, optimal_cost as get_optimal_cost


def set_features(G):
    for e in G.edges:
        i, j = e

        G.edges[e]['features'] = np.array([
            G.edges[e]['weight'],
        ], dtype=np.float32)


def set_labels(G):
    optimal_cost = get_optimal_cost(G)

    for e in G.edges:
        regret = 0.

        if not G.edges[e]['in_solution']:
            tour = fixed_edge_tour(G, e, scale=1e6, max_trials=100, runs=10)
            cost = tour_cost(G, tour)
            regret = (cost - optimal_cost) / optimal_cost

        G.edges[e]['regret'] = regret


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, instances_file, scalers_file=None, feat_drop_