import itertools

import numpy as np


def two_opt(tour, i, j):
    if i == j:
        return tour
    e