
#!/usr/bin/env python
# coding: utf-8

import argparse
import itertools
import multiprocessing as mp
import pathlib
import uuid

import networkx as nx
import numpy as np

import gnngls
from gnngls import datasets


def prepare_instance(G):
    datasets.set_features(G)
    datasets.set_labels(G)
    return G


def get_solved_instances(n_nodes, n_instances):
    for _ in range(n_instances):
        G = nx.Graph()

        coords = np.random.random((n_nodes, 2))
        for n, p in enumerate(coords):
            G.add_node(n, pos=p)

        for i, j in itertools.combinations(G.nodes, 2):
            w = np.linalg.norm(G.nodes[j]['pos'] - G.nodes[i]['pos'])
            G.add_edge(i, j, weight=w)
