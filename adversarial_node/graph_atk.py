import graph_gen as gg
import copy
import numpy as np
import json

def atk_graph(G, p_atk):
  # atk nodes with probability p_atk
  counter = 0
  for node in list(G.nodes):
    if np.random.random() < p_atk:
      G.remove_node(node)
      counter += 1
  return counter

def atk_single_experiment(n, deg, p_mut, p_atk, iters=5):
  avg_cc_list = []
  avg_cp_list = []
  for _i in range(iters):
    G = gg.lattice(n, deg)
    gg.sw_mutate_rem_edges(G, p_mut)
    gg.sw_mutate_add_edges(G)
    orig_cc = gg.clustering_coefficient(G)
    orig_cp = gg.characteristic_path(G)
    for _j in range(iters):
      # atk nodes with probability p_atk
      G_p = copy.deepcopy(G)
      atk_graph(G_p, p_atk)
      new_cc = gg.clustering_coefficient(G_p)
      new_cp = gg.characteristic_path(G_p)
      avg_cc_list.append(new_cc / orig_cc)
      avg_cp_list.append(new_cp / orig_cp)
  avg_clustering = np.mean(avg_cc_list)
  std_clustering = np.std(avg_cc_list)
  avg_cp = np.nanmean(avg_cp_list)
  std_cp = np.std(avg_cp_list)  
  return (avg_clustering, std_clustering, avg_cp, std_cp)

def atk_full_experiment(n, deg, p_mut, start=-3, stop=0, num=31, iters=5):
  cc_data = {"mean":[], "std":[]}
  cp_data = {"mean":[], "std":[]}

  p_space = np.logspace(start, stop, num)
  for p_atk in p_space:
    print(p_atk)
    new_cc, std_cc, new_cp, std_cp = atk_single_experiment(n, deg, p_mut, p_atk, iters)
    cc_data["mean"].append(new_cc)
    cc_data["std"].append(std_cc)
    cp_data["mean"].append(new_cp)
    cp_data["std"].append(std_cp)
  print(cc_data)
  print(cp_data)
  with open('cc_data.txt', 'w') as cc_file, open('cp_data.txt', 'w') as cp_file:
    json.dump(cc_data, cc_file)
    json.dump(cp_data, cp_file)