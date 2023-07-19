import time

import networkx as nx
import numpy as np

from . import tour_cost, operators


def nearest_neighbor(G, depo