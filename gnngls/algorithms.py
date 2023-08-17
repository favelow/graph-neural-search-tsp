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
        i = tour[-1]

        neighbours = [(j, G.edges[(i, j)][guide]) for j in G.neighbors(i) if j not in tour]

        nodes, p = zip(*neighbours)

        p = np.array(p)

        # if there are any infinite values, make these 1 and others 0
        is_inf = np.isinf(p)
        if is_inf.any():
            p = is_inf

        # if there are all 0s, make everything 1
        if np.sum(p) == 0:
            p[:] = 1.

        # if the guide should be inverted, for example, edge weight
        if invert:
            p = 1 / p

        j = np.random.choice(nodes, p=p / np.sum(p))
        tour.append(j)

    tour.append(depot)
    return tour


def best_probabilistic_nearest_neighbour(G, depot, n_iters, guide='weight', weight='weight'):
    best_tour = None
    best_cost = 0

    for _ in range(n_iters):
        new_tour = probabilistic_nearest_neighbour(G, depot, guide)
        new_cost = tour_cost(G, new_tour, weight)

        if new_cost < best_cost or best_tour is None:
            best_tour, best_cost = new_tour, new_cost

    return best_tour


def cheapest_insertion(G, sub_tour, n, weight='weight'):
    best_tour = None
    best_cost = 0

    for j in range(1, len(sub_tour)):
        new_tour = sub_tour.copy()
        new_tour.insert(j, n)
        new_cost = tour_cost(G, new_tour, weight)
