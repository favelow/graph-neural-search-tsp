
import concorde.tsp as concorde
import lkh
import networkx as nx
import numpy as np
import tsplib95
from matplotlib import colors


def tour_to_edge_attribute(G, tour):
    in_tour = {}
    tour_edges = list(zip(tour[:-1], tour[1:]))
    for e in G.edges:
        in_tour[e] = e in tour_edges or tuple(reversed(e)) in tour_edges
    return in_tour


def tour_cost(G, tour, weight='weight'):
    c = 0
    for e in zip(tour[:-1], tour[1:]):
        c += G.edges[e][weight]
    return c


def is_equivalent_tour(tour_a, tour_b):
    if tour_a == tour_b[::-1]:
        return True
    if tour_a == tour_b:
        return True
    return False


def is_valid_tour(G, tour):
    if tour[0] != 0:
        return False
    if tour[-1] != 0:
        return False
    for n in G.nodes:
        c = tour.count(n)