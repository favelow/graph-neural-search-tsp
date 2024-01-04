import itertools

import numpy as np


def two_opt(tour, i, j):
    if i == j:
        return tour
    elif j < i:
        i, j = j, i
    return tour[:i] + tour[j - 1:i - 1:-1] + tour[j:]


def two_opt_cost(tour, D, i, j):
    if i == j:
        return 0
    elif j < i:
        i, j = j, i

    a = tour[i]
    b = tour[i - 1]
    c = tour[j]
    d = tour[j - 1]

    delta = D[a, c] \
            + D[b, d] \
            - D[a, b] \
            - D[c, d]
    return delta


def two_opt_a2a(tour, D, first_improvement=False):
    best_move = None
    best_delta = 0

    idxs = range(1, len(tour) - 1)
    for i, j in itertools.combinations(idxs, 2):
        if abs(i - j) < 2:
            continue

        delta = two_opt_cost(tour, D, i, j)
        if delta < best_delta and not np.isclose(0, delta):
            best_delta = delta
            best_move = i, j
            if first_improvement:
                break

    if best_move is not None:
        return best_delta, two_opt(tour, *best_move)
    return 0, t