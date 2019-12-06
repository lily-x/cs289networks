import matplotlib.pyplot as plt
import numpy as np
import json

# graph temp cc change
def sa_cc_improvement_graph(start=-2, stop=0, num=9):
  p_list = np.logspace(start, stop, num)
  name='sa_cp_improvement.png'

  hot_cc_f = 'sa_0.1perturbation_high_temp/cc_data.txt'
  hot_cp_f = 'sa_0.1perturbation_high_temp/cp_data.txt'
  with open(hot_cc_f, 'r') as cc_file, open(hot_cp_f, 'r') as cp_file:
    hot_cc_data = json.load(cc_file)
    hot_cp_data = json.load(cp_file)

  var_cc_f = 'sa_0.1perturbation_var_temp/cc_data_var_temp.txt'
  var_cp_f = 'sa_0.1perturbation_var_temp/cp_data_var_temp.txt'
  with open(var_cc_f, 'r') as cc_file, open(var_cp_f, 'r') as cp_file:
    var_cc_data = json.load(cc_file)
    var_cp_data = json.load(cp_file)

  cool_cc_f = 'sa_0.1perturbation_low_temp/cc_data_low_temp.txt'
  cool_cp_f = 'sa_0.1perturbation_low_temp/cp_data_low_temp.txt'
  with open(cool_cc_f, 'r') as cc_file, open(cool_cp_f, 'r') as cp_file:
    cool_cc_data = json.load(cc_file)
    cool_cp_data = json.load(cp_file)


  plt.figure()
  plt.errorbar(p_list, hot_cp_data['improvement_mean'], fmt='rx-', yerr=hot_cp_data['improvement_std'], label="Hot SA improvement post-attack", capsize=4)
  plt.errorbar(p_list, var_cp_data['improvement_mean'], fmt='y+-', yerr=var_cp_data['improvement_std'], label="Variable SA improvement post-attack", capsize=4)
  plt.errorbar(p_list, cool_cp_data['improvement_mean'], fmt='gd-', yerr=cool_cp_data['improvement_std'], label="Cool SA improvement post-attack", capsize=4)

  # plt.errorbar(p_list, cp2_data['improvement_mean'], fmt='r^-', yerr=cp_data['improvement_std'], label="cp_exp / cc_0")
  # plt.errorbar(p_list, cp_data['improvement_mean'], fmt='c^--', yerr=cp_data['improvement_std'], label="cp_o / cc_0")

  plt.xscale('log')
  # plt.yscale('log')
  plt.ylabel('$cp_{post\_anneal} / cp_{pre\_anneal}$')
  plt.xlabel('$p_{node\_attack}$')
  plt.title('CP Change At Varying Initial Temperatures')
  plt.legend()
  plt.show()
  plt.savefig(name)
  print("done")

def graph_recovery(start=-2, stop=0, num=9, name='cc_cp_recovery.png'):
  p_list = np.logspace(start, stop, num)
  cc_fname = 'sa_0.1perturbation_high_temp/cc_data.txt'
  cp_fname = 'sa_0.1perturbation_high_temp/cp_data.txt'
  with open(cc_fname, 'r') as cc_file, open(cp_fname, 'r') as cp_file:
    cc_data = json.load(cc_file)
    cp_data = json.load(cp_file)
  plt.figure()
  plt.errorbar(p_list, cc_data['mean'], fmt='bo-', yerr=cc_data['std'], label="$cc_{recovery} / cc_{pre\_attack}$", capsize=4)
  plt.errorbar(p_list, cp_data['mean'], fmt='r^-', yerr=cp_data['std'], label="$cp_{recovery} / cc_{pre\_attack}$", capsize=4)

  plt.xscale('log')
  plt.xlabel('$p_{node\_attack}$')
  plt.ylabel('SW properties proportional to pre-attack sw-network')
  plt.title('SW-Properties Post Attack And Recovery')
  plt.legend()
  plt.show()
  plt.savefig(name)
  print("done")

def sa_full_graph(start=-2, stop=0, num=9):
  p_list = np.logspace(start, stop, num)
  name='sa_cp_improvement_full.png'

  full_cc_f = 'sa_0.1_full_graph_mutation_high_temp/cc_data.txt'
  full_cp_f = 'sa_0.1_full_graph_mutation_high_temp/cp_data.txt'
  with open(full_cc_f, 'r') as cc_file, open(full_cp_f, 'r') as cp_file:
    full_cc_data = json.load(cc_file)
    full_cp_data = json.load(cp_file)

  plt.figure()
  plt.errorbar(p_list, full_cc_data['improvement_mean'], fmt='rx-', yerr=full_cc_data['improvement_std'], label="CC for full graph SA", capsize=4)


  # plt.errorbar(p_list, cp2_data['improvement_mean'], fmt='r^-', yerr=cp_data['improvement_std'], label="cp_exp / cc_0")
  # plt.errorbar(p_list, cp_data['improvement_mean'], fmt='c^--', yerr=cp_data['improvement_std'], label="cp_o / cc_0")

  plt.xscale('log')
  # plt.yscale('log')
  plt.ylabel('$cc_{post\_anneal} / cc_{pre\_anneal}$')
  plt.xlabel('$p_{node\_attack}$')
  plt.title('CC Change For Full Graph Simulated Annealing')
  plt.legend()
  plt.show()
  plt.savefig(name)
  print("done")
