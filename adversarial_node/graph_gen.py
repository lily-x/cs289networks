import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json

def graph(start=-3, stop=0, num=31, name='cc_cp_sw_combined.png'):
  p_list = np.logspace(start, stop, num)
  cc_fname = 'sw_250_4dlimit/cc_data_250_4_20.txt'
  cp_fname = 'sw_250_4dlimit/cp_data_250_4_20.txt'
  with open(cc_fname, 'r') as cc_file, open(cp_fname, 'r') as cp_file:
    cc_data = json.load(cc_file)
    cp_data = json.load(cp_file)

  cc2_fname = 'sw_250_classic/cc_data.txt'
  cp2_fname = 'sw_250_classic/cp_data.txt'
  with open(cc2_fname, 'r') as cc2_file, open(cp2_fname, 'r') as cp2_file:
    cc2_data = json.load(cc2_file)
    cp2_data = json.load(cp2_file)
  plt.figure()
  plt.errorbar(p_list, cc2_data['mean'], fmt='bo-', yerr=cc_data['std'], label="$cc_{deg\_limit} / cc_0$", capsize=4)
  plt.errorbar(p_list, cp2_data['mean'], fmt='r^-', yerr=cp_data['std'], label="$cp_{deg\_limit} / cc_0$", capsize=4)
  plt.errorbar(p_list, cc_data['mean'], fmt='ko--', yerr=cc_data['std'], label="$cc_{original} / cc_0$", capsize=4)
  plt.errorbar(p_list, cp_data['mean'], fmt='c^--', yerr=cp_data['std'], label="$cp_{original} / cc_0$", capsize=4)
  plt.xscale('log')
  plt.xlabel('$p_{edge\_mutation}$')
  plt.ylabel('Change proportional to starting lattice')
  plt.title('Degree Limited Small-World Network Formation')
  plt.legend()
  plt.show()
  plt.savefig(name)
  print("done")

def lattice(n, deg):
  G = nx.Graph(deg=deg)
  G.add_nodes_from(range(n))
  for node in range(n):
    vertices = range(node - deg // 2, node + deg // 2 + 1)
    edges = [(node, v % n) for v in vertices if v != node]
    G.add_edges_from(edges)
  return G

def clustering_coefficient(G):
  local_coeffs = []
  for node in list(G.nodes):
    neighbors = list(G.adj[node])
    if len(neighbors) < 2:
      continue
    cnxns = 0
    possible = len(neighbors) * (len(neighbors) - 1) / 2
    for a, b in list(G.edges):
      if a in neighbors and b in neighbors:
        cnxns += 1
    local_coeffs.append(cnxns / possible)
  return np.mean(local_coeffs)

def characteristic_path(G):
  try:
    return nx.average_shortest_path_length(G)
  except:
    return np.nan 

# classic sw formation
def sw_classic_mutate(G, p):
  remaining_edges = []
  counter = 0
  nodes = list(G.nodes)
  for a, b in list(G.edges):
    if np.random.random() < p:
      G.remove_edge(a, b)
      counter += 1
    else:
      remaining_edges.append((a, b))
  
  for _i in range(counter):
    found = False
    while not found:
      [a, b] = np.random.choice(nodes, 2)
      found = not G.has_edge(a, b)
    G.add_edge(a, b)
    remaining_edges.append((a, b))
  # degs = [G.degree(node) for node in nodes]
  # print(max(degs))
  return remaining_edges
  # print(f"Removed {counter} edges")

# Probabilistically remove edges from 'edges'
def sw_mutate_rem_edges(G, p, edges = None):
  if not edges:
    edges = list(G.edges)

  remaining_edges = []
  counter = 0
  for a, b in edges:
    if np.random.random() < p:
      G.remove_edge(a, b)
      counter += 1
    else:
      remaining_edges.append((a, b))
  return remaining_edges
  # print(f"Removed {counter} edges")

def possible_edge(G, vacancies):
  for i, (n1, _v) in enumerate(vacancies):
    for n2, _v in vacancies[i+1:]:
      if not G.has_edge(n1, n2):
        return True
  return False

def sw_mutate_add_edges(G):
  vacancies = []
  new_edges = []
  for n in list(G.nodes):
    v = G.graph["deg"] - len(list(G.adj[n]))
    if v > 0:
      vacancies.append((n, v))
  counter = 0
  while possible_edge(G, vacancies):
    # print(vacancies)
    chosen = np.random.choice(len(vacancies), 2, replace=False)
    n1, v1 = vacancies[chosen[0]]
    n2, v2 = vacancies[chosen[1]]
    if G.has_edge(n1, n2):
      continue
    vacancies[chosen[0]] = (n1, v1 - 1)
    vacancies[chosen[1]] = (n2, v2 - 1)
    vacancies = [tup for tup in vacancies if tup[1] > 0]
    G.add_edge(n1, n2)
    counter += 1
    new_edges.append((n1, n2))
  return new_edges
  # print(f"Added {counter} edges")
  # print(vacancies)
  

def sw_single_experiment(n, deg, p, iters=5):
  G = lattice(n, deg)
  original_clustering = clustering_coefficient(G)
  original_cp = characteristic_path(G)

  # print(f"original clustering {original_clustering}")
  # print(f"original cp {original_cp}")
  new_clustering = []
  new_cp = []
  for i in range(iters):
    sw_mutate_rem_edges(G, p)
    sw_mutate_add_edges(G)
    new_clustering.append(clustering_coefficient(G))
    new_cp.append(characteristic_path(G))
    G = lattice(n, deg)
  avg_clustering = np.mean(new_clustering)/original_clustering
  std_clustering = np.std(new_clustering)/original_clustering
  avg_cp = np.nanmean(new_cp)/original_cp
  std_cp = np.std(new_cp)/original_cp
  # print(f"Avg clustering coefficient: {avg_clustering}\n{avg_clustering/original_clustering}")
  # print(f"Avg cp: {avg_cp}\n{avg_cp/original_cp}")
  return (avg_clustering, std_clustering, avg_cp, std_cp)

def sw_full_experiment(n, deg, start=-3, stop=0, num=31, iters=20):
  cc_data = {"mean":[], "std":[]}
  cp_data = {"mean":[], "std":[]}

  p_space = np.logspace(start, stop, num)
  for p in p_space:
    print(p)
    new_cc, std_cc, new_cp, std_cp = sw_single_experiment(n, deg, p, iters)
    cc_data["mean"].append(new_cc)
    cc_data["std"].append(std_cc)
    cp_data["mean"].append(new_cp)
    cp_data["std"].append(std_cp)
  print(cc_data)
  print(cp_data)
  with open('cc_data.txt', 'w') as cc_file, open('cp_data.txt', 'w') as cp_file:
    json.dump(cc_data, cc_file)
    json.dump(cp_data, cp_file)

def sw_single_experiment_classic(n, deg, p, iters=5):
  G = lattice(n, deg)
  original_clustering = clustering_coefficient(G)
  original_cp = characteristic_path(G)

  # print(f"original clustering {original_clustering}")
  # print(f"original cp {original_cp}")
  new_clustering = []
  new_cp = []
  for i in range(iters):
    sw_classic_mutate(G, p)
    new_clustering.append(clustering_coefficient(G))
    new_cp.append(characteristic_path(G))
    G = lattice(n, deg)
  avg_clustering = np.mean(new_clustering)/original_clustering
  std_clustering = np.std(new_clustering)/original_clustering
  avg_cp = np.nanmean(new_cp)/original_cp
  std_cp = np.std(new_cp)/original_cp
  # print(f"Avg clustering coefficient: {avg_clustering}\n{avg_clustering/original_clustering}")
  # print(f"Avg cp: {avg_cp}\n{avg_cp/original_cp}")
  return (avg_clustering, std_clustering, avg_cp, std_cp)

def sw_full_experiment_classic(n, deg, start=-3, stop=0, num=31, iters=20):
  cc_data = {"mean":[], "std":[]}
  cp_data = {"mean":[], "std":[]}

  p_space = np.logspace(start, stop, num)
  for p in p_space:
    print(p)
    new_cc, std_cc, new_cp, std_cp = sw_single_experiment_classic(n, deg, p, iters)
    cc_data["mean"].append(new_cc)
    cc_data["std"].append(std_cc)
    cp_data["mean"].append(new_cp)
    cp_data["std"].append(std_cp)
  print(cc_data)
  print(cp_data)
  with open('cc_data.txt', 'w') as cc_file, open('cp_data.txt', 'w') as cp_file:
    json.dump(cc_data, cc_file)
    json.dump(cp_data, cp_file)

def sw_full_experiment_deg(n, p_mut=0.1, start=-3, stop=0, num=31, iters=20):
  cc_data = {"mean":[], "std":[]}
  cp_data = {"mean":[], "std":[]}

  for deg in range(2, 51, 2):
    print(deg)
    new_cc, std_cc, new_cp, std_cp = sw_single_experiment(n, 2*deg, p_mut, iters)
    cc_data["mean"].append(new_cc)
    cc_data["std"].append(std_cc)
    cp_data["mean"].append(new_cp)
    cp_data["std"].append(std_cp)
  print(cc_data)
  print(cp_data)
  with open('sw_deg_cc_data.txt', 'w') as cc_file, open('sw_deg_cp_data.txt', 'w') as cp_file:
    json.dump(cc_data, cc_file)
    json.dump(cp_data, cp_file)

def graph_deg(start=2, stop=100, num=50, name='deg_cc_cp_plot.png', cc_fname='sw_deg_250_4_0.1/sw_deg_cc_data.txt', cp_fname='sw_deg_250_4_0.1/sw_deg_cp_data.txt'):
  xs = range(1, 51, 2)
  with open(cc_fname, 'r') as cc_file, open(cp_fname, 'r') as cp_file:
    cc_data = json.load(cc_file)
    cp_data = json.load(cp_file)
  plt.figure()
  plt.errorbar(xs, cc_data['mean'], fmt='bo-', yerr=cc_data['std'], label='$cc/cc_0$', capsize=4)
  plt.errorbar(xs, cp_data['mean'], fmt='r^-', yerr=cp_data['std'], label='$cp/cp_0$', capsize=4)
  plt.xlabel('maximal node degree')
  plt.ylabel('Change proportional to starting lattice')
  plt.title('Small-World Properties At Varying Node Degrees')
  plt.legend()
  plt.show()
  plt.savefig(name)
  print("done")

def graph_atk(start=-3, stop=0, num=31, name='cc_cp_post_atk.png'):
  p_list = np.logspace(start, stop, num)
  cc_fname = 'atk_p_atk_0.1/cc_data_250_4_5_0.1.txt'
  cp_fname = 'atk_p_atk_0.1/cp_data_250_4_5_0.1.txt'
  with open(cc_fname, 'r') as cc_file, open(cp_fname, 'r') as cp_file:
    cc_data = json.load(cc_file)
    cp_data = json.load(cp_file)

  plt.figure()
  plt.errorbar(p_list, cc_data['mean'], fmt='bo-', yerr=cc_data['std'], label="$cc_{post\_attack} / cc_{sw}$", capsize=4)
  plt.errorbar(p_list, cp_data['mean'], fmt='r^-', yerr=cp_data['std'], label="$cp_{post\_attack} / cc_{sw}$", capsize=4)
  plt.xscale('log')
  plt.xlabel('$p_{node\_attack}$')
  plt.ylabel('Change proportional to pre-attack small-worlds network')
  plt.title('Post-Node-Attack Small-World Properties')
  plt.legend()
  plt.show()
  plt.savefig(name)
  print("done")
  