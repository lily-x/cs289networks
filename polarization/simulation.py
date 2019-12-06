import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation

SEED = 100
random.seed(SEED)
np.random.seed(SEED)

class Polarization:
    def __init__(self, n, m, p, graph=None):
        if graph is not None:
            self.graph = graph
            self.n = n
            self.m = m
        else:
            graph = nx.algorithms.bipartite.generators.random_graph(n,m,p)
            for i in range(n):
                graph.node[i]['value'] = np.random.uniform(-1,1)

            for i in range(n):
                graph.node[i]['gamma'] = 0.05 # np.random.uniform(0.0,0.1)

            self.graph = graph
            self.n = n
            self.m = m
            self.p = p

            graph.node[0]['value'] = 1
            graph.node[0]['gamma'] = 0.05


        self.bar_aggregation = np.average
        # top = nx.bipartite.sets(self.graph)[0]
        # pos = nx.bipartite_layout(self.graph, top)
        # nx.draw(self.graph, pos=pos)  # use spring layout
        # plt.show()

        self.fig, self.ax = plt.subplots(figsize=(10,10))
        top = nx.bipartite.sets(self.graph)[0]
        self.pos = nx.bipartite_layout(self.graph, top)


    def simulate(self, num):
        self.ax.clear()

        total_polarization = 0
        red_edgelist = []
        red_widthlist = []
        mapping = {}
        # i refers to the left node, j refers to the right node
        for i in range(self.n): # randomly assign a bar to go,
            neighbors = list(self.graph.neighbors(i))
            if len(neighbors) == 0:
                continue
            for j in neighbors: # initialization
                self.graph.edges[i,j]['prob'] = 0
            j = np.random.choice(neighbors)
            red_edgelist.append((i,j))
            mapping[i] = j

            prob = 1
            self.graph.edges[i,j]['prob'] = prob
            red_widthlist.append(prob)
            total_polarization += self.graph.node[i]['value']
            # print('polarization:', self.graph.node[i]['value'])

        total_polarization /= self.n
        print('Total polarization: {}'.format(total_polarization))

        for j in range(self.n,self.n+self.m): # bar aggregation
            values = [self.graph.node[i]['value'] for i in self.graph.neighbors(j)]
            # weights = [self.graph.edges[i,j]['prob'] for i in self.graph.neighbors(j)]
            weights = [self.graph.edges[i,j]['prob'] * self.graph.node[i]['gamma'] for i in self.graph.neighbors(j)]
            if sum(weights) == 0:
                self.graph.node[j]['value'] = 0
                continue
            else:
                self.graph.node[j]['value'] = self.bar_aggregation(values, weights=weights)

        for i in range(self.n): # people aggregation
            j = mapping[i]
            self.graph.node[i]['value'] = self.graph.node[i]['value'] * (1 - self.graph.node[i]['gamma']) + self.graph.node[j]['value'] * self.graph.node[i]['gamma']

        node_color = [self.graph.node[i]['value'] for i in range(self.n+self.m)]
        nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=self.graph.edges, width=0.5, ax=self.ax, edge_color='grey', style='dotted')

        nx.draw_networkx_nodes(self.graph, pos=self.pos, nodelist=self.graph.nodes, ax=self.ax, node_color=node_color, cmap=plt.cm.coolwarm, vmin=-1, vmax=1)
        nx.draw_networkx_edges(self.graph, pos=self.pos, edgelist=red_edgelist, width=0.5, ax=self.ax, edge_color='grey')

        self.ax.set_title("Iteration %d    "%(num+1) +  ", average polarization: %f"%(total_polarization), fontweight="bold")

        # nx.draw(self.graph, pos=self.pos, ax=self.ax, node_color=node_color, cmap=plt.cm.Blues)  # use spring layout
        # plt.show()


if __name__ == '__main__':
    n, m, p = 10, 5, 0.5
    model = Polarization(n,m,p)
    # model.simulate()

    ani = matplotlib.animation.FuncAnimation(model.fig, model.simulate, frames=100, interval=100, repeat=True)
    ani.save('fixed_attack.gif', writer='imagemagick', fps=5)
    # plt.show()

    
