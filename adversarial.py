""" model a simple adversary
Lily Xu
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random


def get_spectral_gap(g):
    """ algebraic connectivity: second-smallest eigenvalue """
    spectrum = nx.spectrum.normalized_laplacian_spectrum(g)
    spectrum.sort()
    return spectrum[1]


def disconnect_highest_deg(G, num_remove):
    """ disconnect highest-degree neighbor """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    vs = np.random.choice(list(g.nodes()), num_remove, replace=False)
    for i, v in enumerate(vs):
        neighbors = list(g.neighbors(v))
        if len(neighbors) == 0:
            continue
        degrees = np.array([g.degree(n) for n in neighbors])
        remove = np.argmax(degrees)
        g.remove_edge(v, neighbors[remove])

        num_removed.append(i)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap


def disconnect_2highest_deg(G, num_remove):
    """ disconnect neighbor with the highest second degree """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    vs = np.random.choice(list(g.nodes()), num_remove, replace=False)
    for i, v in enumerate(vs):
        neighbors = list(g.neighbors(v))
        if len(neighbors) == 0:
            continue
        max_degree = []
        for n in neighbors:
            nneigh = g.neighbors(n)
            degrees = np.array([g.degree(n) for n in nneigh])
            max_degree.append(np.argmax(degrees))
        remove = np.argmax(max_degree)
        g.remove_edge(v, neighbors[remove])

        num_removed.append(i)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap


def disconnect_lowest_deg(G, num_remove):
    """ disconnect lowest-degree neighbor """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    vs = np.random.choice(list(g.nodes()), num_remove, replace=False)
    for i, v in enumerate(vs):
        neighbors = list(g.neighbors(v))
        if len(neighbors) == 0:
            continue
        degrees = np.array([g.degree(n) for n in neighbors])
        remove = np.argmin(degrees)
        g.remove_edge(v, neighbors[remove])

        num_removed.append(i)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap


def disconnect_lowest_ecc(G, num_remove):
    """ disconnect lowest eccentricity neighbor """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    vs = np.random.choice(list(g.nodes()), num_remove, replace=False)
    for i, v in enumerate(vs):
        neighbors = list(g.neighbors(v))
        if len(neighbors) == 0:
            continue
        ecc = np.array([nx.eccentricity(G, n) for n in neighbors])
        remove = np.argmin(ecc)
        g.remove_edge(v, neighbors[remove])

        num_removed.append(i)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap


def disconnect_random(G, num_remove):
    """ disconnect random edges """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    edges = list(g.edges())
    idxs = np.random.choice(len(edges), num_remove, replace=False)
    for i, idx in enumerate(idxs):
        g.remove_edge(*edges[i])

        num_removed.append(i)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap


def disconnect_random_neigh(G, num_remove):
    """ disconnect random neighbor """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    vs = np.random.choice(list(g.nodes()), num_remove, replace=False)
    for i, v in enumerate(vs):
        neighbors = list(g.neighbors(v))
        if len(neighbors) == 0:
            continue
        remove = np.random.choice(neighbors)
        g.remove_edge(v, remove)

        num_removed.append(i)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap


def disconnect_sum(G, num_remove):
    """ disconnect between nodes with large sum of degrees """
    num_removed = []
    spectral_gap = []

    g = G.copy()
    edges = list(g.edges())
    sum = np.zeros(len(edges))
    for i, (u, v) in enumerate(edges):
        sum[i] = g.degree(u) + g.degree(v)

    idxs = sum.argsort()[-num_remove:][::-1]
    for i, idx in enumerate(idxs):
        g.remove_edge(*edges[idx])
        num_removed.append(i+1)
        spectral_gap.append(get_spectral_gap(g))

    return num_removed, spectral_gap




if __name__ == '__main__':
    num_remove = 50  # number of edges we can remove

    n = 100  # number of nodes
    m = 5   # degree of node
    p = .8  # probability of rewiring

    G = nx.powerlaw_cluster_graph(n, m, p)

    # nx.draw(G, with_labels=True)
    # plt.show()

    gap_orig = get_spectral_gap(G)

    num_removed_sum, spectral_gap_sum = disconnect_sum(G, num_remove)
    num_removed_random_neigh, spectral_gap_random_neigh = disconnect_random_neigh(G, num_remove)
    num_removed_random, spectral_gap_random = disconnect_random(G, num_remove)
    num_removed_lowest_ecc, spectral_gap_lowest_ecc = disconnect_lowest_ecc(G, num_remove)
    num_removed_highest, spectral_gap_highest = disconnect_highest_deg(G, num_remove)
    num_removed_lowest, spectral_gap_lowest = disconnect_lowest_deg(G, num_remove)
    num_removed_2highest, spectral_gap_2highest = disconnect_2highest_deg(G, num_remove)


    plt.figure(figsize=(8,4))
    plt.axhline(y=gap_orig, color='gray', linestyle='-', linewidth=.5, label='orig')
    plt.plot(num_removed_sum, spectral_gap_sum, label='edge sum')
    plt.plot(num_removed_random_neigh, spectral_gap_random_neigh, label='random neigh')
    # plt.plot(num_removed_random, spectral_gap_random, label='random')
    plt.plot(num_removed_lowest_ecc, spectral_gap_lowest_ecc, label='lowest ecc')
    plt.plot(num_removed_highest, spectral_gap_highest, label='highest deg')
    # plt.plot(num_removed_lowest, spectral_gap_lowest, label='lowest deg')
    plt.plot(num_removed_2highest, spectral_gap_2highest, label='highest 2nd degree')
    plt.ylabel('$\lambda_2$')
    plt.title('Adversarial case')
    plt.legend()
    plt.show()

    # rewire
