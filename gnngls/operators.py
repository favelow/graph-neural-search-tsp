import itertools

import numpy as np


def two_opt(tour, i, j):
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j - 1:i - 1:-1] + to