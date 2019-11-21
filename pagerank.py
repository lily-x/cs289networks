import networkx as nx
import numpy as np
import random 

def push(g, u, p, r, alpha):
    p[u] = p[u] + alpha*r[u]
    for w in g.neighbors(u):
        r[w] = r[w] + (1 - alpha)*r[u]/(2*g.degree(u))
    r[u] = (1 - alpha)*r[u]/2
    

def apr(g, v, eps, alpha, rounding = False, num_round_points = 20):
    '''
    Approximate pagerank vector on graph g for node v, with error eps and
    pagerank parameter alpha
    '''
    p = {}
    r = {}
    for u in g:
        p[u] = 0
        r[u] = 0
    r[v] = 1
    updated = True
    while updated:
        updated = False
        for u in g:
            if r[u]/g.degree(u) > eps:
                push(g, u, p, r, alpha)
                updated = True
    q = {v: p[v]/g.degree(v) for v in g}
    if rounding:
        max_q = max(q[v] for v in g)
        min_q = min(q[v] for v in g)
        points = np.linspace(min_q, max_q, num=num_round_points)
        for v in q:
            q[v] = points[np.abs(points - q[v]).argmin()]
    return np.array([q[v] for v in g])

def cut(q, degrees, alpha, phi):
    order = np.argsort(-q)
    q = q[order]
    vol = np.cumsum(degrees[order])
    for i in range(1, len(q)):
        if vol[i] > (1 - phi)*vol[i-1]:
            if q[i] < q[i-1] - alpha/(phi*vol[i-1]):
                return q[i-1], i
    return q[-1], 'end'
                
def gen_sbm(n, p_w, p_b):
    A = np.zeros((n, n))
    block = np.zeros(n)
    block[int(n/2):] = 1
    for i in range(n):
        for j in range(i):
            if block[i] == block[j]:
                if random.random() < p_w:
                    A[i,j] = 1
                    A[j,i] = 1
            else:
                if random.random() < p_b:
                    A[i,j] = 1
                    A[j,i] = 1
    return nx.from_numpy_array(A)


def rewire_network(g, phi, alpha):
    degrees = np.array([g.degree(v) for v in g])
    num_added = []
    spectral_gap = []
    all_added = []
    num_across = 0
    for i in range(100):
        v = random.choice(list(g.nodes()))
        q = apr(g, v, 0.0001, 0.01, rounding=False, num_round_points=100)
        cutoff, i = cut(q, degrees, alpha, phi)
        if i != 'end':
            to_add = []
            for u in g:
                if q[u] >= cutoff:
                    for w in g.neighbors(u):
                        if q[w] <  cutoff:
                            s = v
                            while True:
                                s = random.choice(list(g.neighbors(s)))
                                if random.random() < alpha:
                                    break
                            if q[s] >= cutoff and not g.has_edge(s, w):
                                to_add.append((s, w))
                                all_added.append((s, w))
                                if s < 50 and w >= 50 or w < 50 and s >= 50:
                                    num_across += 1
            g.add_edges_from(to_add)
            num_added.append(len(to_add))
            spectrum = nx.spectrum.normalized_laplacian_spectrum(g)
            spectrum.sort()
            spectral_gap.append(spectrum[1])
    g.remove_edges_from(all_added)    
    print(num_across/len(all_added))
    return num_added, spectral_gap

def random_addition(g):
    all_added = []
    spectral_gap = []
    num_added = []
    num_across = 0
    for i in range(150):
        u = random.choice(list(g.nodes()))
        v = random.choice(list(g.nodes()))
        if not g.has_edge(u,v):
            g.add_edge(u,v)
            if u < 50 and v >= 50 or v < 50 and u >= 50:
                num_across += 1
            num_added.append(1)
            all_added.append((u,v))
            spectrum = nx.spectrum.normalized_laplacian_spectrum(g)
            spectrum.sort()
            spectral_gap.append(spectrum[1])
    g.remove_edges_from(all_added)
    print(num_across/len(all_added))
    return num_added, spectral_gap

def random_local_addition(g, alpha):
    all_added = []
    spectral_gap = []
    num_added = []
    num_across = 0
    for i in range(150):
        v = random.choice(list(g.nodes()))
        u = v
        while True:
            u = random.choice(list(g.neighbors(u)))
            if random.random() < alpha:
                break
        if not g.has_edge(u,v):
            g.add_edge(u,v)
            if u < 50 and v >= 50 or v < 50 and u >= 50:
                num_across += 1
            num_added.append(1)
            all_added.append((u,v))
            spectrum = nx.spectrum.normalized_laplacian_spectrum(g)
            spectrum.sort()
            spectral_gap.append(spectrum[1])
    g.remove_edges_from(all_added)
    print(num_across/len(all_added))
    return num_added, spectral_gap


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



plt.plot(np.cumsum(num_added), spectral_gap); plt.plot(np.cumsum(num_added_random), spectral_gap_random)
plt.xlabel('Number of edges added', fontsize=15)
plt.ylabel('$\lambda_2$', fontsize=20)
plt.legend(['PageRank', 'RandomWalk'], fontsize=15)