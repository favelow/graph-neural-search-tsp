import time

import networkx as nx
import numpy as np

from . import tour_cost, operators


def nearest_neighbor(G, depot, weight='weight'):
    tour = [depot]
    while len(tour) < len(G.nodes):
        i = tour[-1]
        neighbours = [(j, G.edges[(i, j)][weight]) for j in G.neighbors(i) if j not in tour]
        j, dist = min(neighbours, key=lambda e: e[1])
        tour.append(j)

    tour.append(depot)
    return tour


def probabilistic_nearest_neighbour(G, depot, guide='weight', invert=True):
    tour = [depot]

    while len(tour) < len(G.nodes):
 