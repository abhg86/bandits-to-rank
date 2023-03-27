#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""

def dictionnary_for_permutations():
    from collections import defaultdict
    import numpy as np

    nb_arms = 3
    nb_positions = 2
    p = 0.5

    mem = defaultdict(lambda: {'N': 0, 'S': 0, 'R': 0, 'mu_hat': 0})

    print(mem[0])  # => default value

    propositions = np.arange(nb_arms)
    print(propositions)
    # print(mem[propositions]) # => error (non-hashable key)

    print(tuple(propositions))
    print(mem[tuple(propositions)])  # => default value

    for _ in range (10):
        positions = np.arange(nb_arms)
        np.random.shuffle(positions)
        nb_c = np.random.binomial(nb_positions, p)
        r = np.random.binomial(1, nb_c/nb_positions)
        print(positions, nb_c, r)
        positions = tuple(positions)
        stats = mem[positions]
        stats['N'] += 1
        stats['S'] += r
        stats['R'] += nb_c
        stats['mu_hat'] = stats['R'] / stats['N']
        mem[positions] = stats
    for keys, values in mem.items():
        print(keys, values)


if __name__ == "__main__":
    dictionnary_for_permutations()
