import networkx as nx
import matplotlib.pyplot as plt
import graph_gen as gg
from graph_atk import atk_graph
import copy
import numpy as np
import json
import math

def fitness(G, cc_0, cp_0):
  cc_1 = gg.clustering_coefficient(G) 
  # cp_1 = gg.characteristic_path(G)
  connected_components = len(list(nx.connected_components(G)))
  # clustering_score = 10 * (cc_1 / cc_0)
  # path_score = (cp_0 - cp_1) / cp_0
  
  # testing shows path_score was still weighted too strongly
  # much harder to improve clustering score than path score
  path_score = 0
  clustering_score = cc_1

  cnxn_score = 0.5 ** (connected_components - 1)
  # score of original graph = 3
  return cnxn_score * (clustering_score + path_score)

def add_nodes(G, n_recover):
  nodes = sorted(list(G.nodes))
  if not nodes:
    cur = 1
  else:
    cur = sorted(list(G.nodes))[-1] + 1
  G.add_nodes_from(range(cur, cur + n_recover))
  return gg.sw_mutate_add_edges(G)

def mutate_new_edges(G, p, new_edges):
  remaining_edges = gg.sw_mutate_rem_edges(G, p, new_edges)
  new_edges = gg.sw_mutate_add_edges(G)
  return remaining_edges + new_edges

def fitness_single_test(n, deg, p_mut, p_atk, n_recover = None, iters=1):
  avg_cc_list = []
  avg_cp_list = []
  avg_fit_list = []
  for _i in range(iters):
    G = gg.lattice(n, deg)
    gg.sw_mutate_rem_edges(G, p_mut)
    gg.sw_mutate_add_edges(G)
    orig_cc = gg.clustering_coefficient(G)
    orig_cp = gg.characteristic_path(G)
    for _j in range(iters):
      # atk nodes with probability p_atk
      G_p = copy.deepcopy(G)
      recover = atk_graph(G_p, p_atk)
      if n_recover is not None:
        recover = n_recover(recover)

      # add new nodes
      add_nodes(G_p, recover)

      # check fitness post prelim recovery
      new_cc = gg.clustering_coefficient(G_p)
      new_cp = gg.characteristic_path(G_p)
      avg_cc_list.append(new_cc / orig_cc)
      avg_cp_list.append(new_cp / orig_cp)
      avg_fit_list.append(fitness(G_p, orig_cc, orig_cp))
  avg_clustering = np.mean(avg_cc_list)
  std_clustering = np.std(avg_cc_list)
  avg_cp = np.nanmean(avg_cp_list)
  std_cp = np.std(avg_cp_list)
  avg_fit = np.nanmean(avg_fit_list)
  std_fit = np.std(avg_fit_list)
  return (avg_clustering, std_clustering, avg_cp, std_cp, avg_fit, std_fit)

def fitness_full_experiment(n, deg, p_mut, start=-3, stop=0, num=31, iters=5, n_recover=None):
  cc_data = {"mean":[], "std":[]}
  cp_data = {"mean":[], "std":[]}
  fit_data = {"mean":[], "std":[]}

  p_space = np.logspace(start, stop, num)
  for p_atk in p_space:
    print(p_atk)
    new_cc, std_cc, new_cp, std_cp, new_fit, std_fit = fitness_single_test(n, deg, p_mut, p_atk, iters=iters, n_recover=n_recover)
    cc_data["mean"].append(new_cc)
    cc_data["std"].append(std_cc)
    cp_data["mean"].append(new_cp)
    cp_data["std"].append(std_cp)
    fit_data["mean"].append(new_fit)
    fit_data["std"].append(std_fit)
  print(cc_data)
  print(cp_data)
  print(fit_data)
  with open('cc_data.txt', 'w') as cc_file, open('cp_data.txt', 'w') as cp_file, open('fit_data.txt', 'w') as fit_file:
    json.dump(cc_data, cc_file)
    json.dump(cp_data, cp_file)
    json.dump(fit_data, fit_file)

def graph_fitness(start=-3, stop=0, num=31, name='fit_plot.png', fit_fname='fit_data.txt', cp_fname='fit_data.txt'):
  p_list = np.logspace(start, stop, num)
  with open(fit_fname, 'r') as fit_file, open(cp_fname, 'r') as cp_file:
    fit_data = json.load(fit_file)
  plt.figure()
  plt.errorbar(p_list, fit_data['mean'], fmt='ks-', yerr=fit_data['std'])
  plt.xscale('log')
  plt.show()
  plt.savefig(name)
  print("done")

def simulate_annealing(G, cc_0, cp_0, new_edges, p_mut, step=0.02, threshold=0.005):
  fitness_1 = fitness(G, cc_0 ,cp_0)
  # experiments for altering starting temp based on fitness
  # temp = fitness_1 / fitness_0
  # if temp < 0.85:
  #   temp = 2
  # elif temp > 1.15:
  #   temp = -2
  # else:
  #   temp = 0

  temp = 2
  change_prob = 1 / (1 + math.exp(-temp))
  steps = 0
  while(change_prob > threshold):
    # print(steps, temp, change_prob)
    if(steps % 50 == 0):
      print(steps)
    cur_fitness = fitness(G, cc_0, cp_0)
    G_p = copy.deepcopy(G)
    newest_edges = mutate_new_edges(G_p, p_mut, new_edges)
    new_fitness = fitness(G_p, cc_0, cp_0)
    if np.random.random() < change_prob or new_fitness > cur_fitness:
      G = G_p
      new_edges = newest_edges
      
    temp -= step
    change_prob = 1 / (1 + math.exp(-temp))
    steps += 1
    # print(fitness(G, cc_0, cp_0), gg.clustering_coefficient(G)/cc_0, gg.characteristic_path(G)/cp_0)
    # print("\n")

  return G, steps
  

def sg_simulated_annealing_recovery(p_atk=0.1, p_mut=0.05, n_recover=None, iters=2):
  ccs = []
  cps = []
  cc_improvement = []
  cp_improvement = []

  for _x in range(iters):
    G = gg.lattice(250, 4)
    gg.sw_mutate_rem_edges(G, p_mut)
    gg.sw_mutate_add_edges(G)
    # calculate original cc and cp
    cc_0 = gg.clustering_coefficient(G)
    cp_0 = gg.characteristic_path(G)
    for _i in range(iters):
      G_p = copy.deepcopy(G)
      recover = atk_graph(G_p, p_atk)
      if recover == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
      print(f"recover is {recover}")
      if n_recover is not None:
        recover = n_recover(recover)

      for _j in range(iters):
        # add new nodes
        G_a = copy.deepcopy(G_p)
        new_edges = add_nodes(G_a, recover)
        cc_comp = gg.clustering_coefficient(G_a)
        cp_comp = gg.characteristic_path(G_a)
        G_a, step_count = simulate_annealing(G_a, cc_0, cp_0, new_edges, p_mut)
        ccs.append(gg.clustering_coefficient(G_a) / cc_0)
        cps.append(gg.characteristic_path(G_a) / cp_0)
        cc_improvement.append(gg.clustering_coefficient(G_a) / cc_comp)
        cp_improvement.append(gg.characteristic_path(G_a) / cp_comp)
  
  avg_clustering = np.mean(ccs)
  std_clustering = np.std(ccs)
  avg_cp = np.nanmean(cps)
  std_cp = np.std(cps)
  avg_cc_imp = np.nanmean(cc_improvement)
  std_cc_imp = np.std(cc_improvement)
  avg_cp_imp = np.nanmean(cp_improvement)
  std_cp_imp = np.std(cp_improvement)
  return (avg_clustering, std_clustering, avg_cp, std_cp, avg_cc_imp, std_cc_imp, avg_cp_imp, std_cp_imp)

def sa_full_experiment(start=-2, stop=0, num=9, iters=2):
  cc_data = {"mean":[], "std":[], "improvement_mean":[], "improvement_std":[]}
  cp_data = {"mean":[], "std":[], "improvement_mean":[], "improvement_std":[]}

  p_space = np.logspace(start, stop, num)
  print(p_space)
  for p_atk in p_space:
    print(p_atk)
    new_cc, std_cc, new_cp, std_cp, avg_cc_imp, std_cc_imp, avg_cp_imp, std_cp_imp = sg_simulated_annealing_recovery(p_atk=p_atk, iters=iters)
    cc_data["mean"].append(new_cc)
    cc_data["std"].append(std_cc)
    cc_data["improvement_mean"].append(avg_cc_imp)
    cc_data["improvement_std"].append(std_cc_imp)
    cp_data["mean"].append(new_cp)
    cp_data["std"].append(std_cp)
    cp_data["improvement_mean"].append(avg_cp_imp)
    cp_data["improvement_std"].append(std_cp_imp)
  print(cc_data)
  print(cp_data)
  with open('cc_data.txt', 'w') as cc_file, open('cp_data.txt', 'w') as cp_file:
    json.dump(cc_data, cc_file)
    json.dump(cp_data, cp_file)

def graph_improvement(start=-2, stop=0, num=9, name='cc_cp_improvement_plot.png', cc_fname='cc_data.txt', cp_fname='cp_data.txt'):
  p_list = np.logspace(start, stop, num)
  with open(cc_fname, 'r') as cc_file, open(cp_fname, 'r') as cp_file:
    cc_data = json.load(cc_file)
    cp_data = json.load(cp_file)
  plt.figure()
  plt.errorbar(p_list, cc_data['improvement_mean'], fmt='bo-', yerr=cc_data['improvement_std'])
  plt.errorbar(p_list, cp_data['improvement_mean'], fmt='r^-', yerr=cp_data['improvement_std'])
  plt.xscale('log')
  plt.show()
  plt.savefig(name)
  print("done")