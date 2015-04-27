import numpy as np
import numpy.random as npr
import networkx as nx
import operator
import itertools
import collections
import random
import time
import cPickle
import matplotlib.pyplot as plt
import gzip

"""
Not sure why PBM not working
The thing to do, then is create an utterly conventional BM with boring kids topology
and only then make it more interesting with the cool kids topology
"""

def create_complete_graph(n=10):
    net = nx.complete_graph(n)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 0.5
    for node in net.nodes_iter():
        net.node[node]["state"] = int(round(random.random()))
    return net

def load_krongraph(filename="kron.edgelist"):
    net = nx.read_edgelist(filename)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 0.5
    for node in net.nodes_iter():
        net.node[node]["state"] = int(round(random.random()))
    return net

def get_seeds(net, num_seeds):
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:num_seeds]
    return [x[0] for x in top]

def get_tops(net, num_seeds):
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:num_seeds]
    nodes = dict(net.nodes(data=True))
    res = []
    for idx, seed in enumerate(top):
        res.append(nodes[seed[0]]["state"])
    return res

def energy(net):
    total_energy = 0
    nodes = dict(net.nodes(data=True))
    for first, second in net.edges_iter():
        weight = net[first][second]["weight"]
        total_energy -= weight * nodes[first]["state"] * nodes[second]["state"]
    return total_energy

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def randomize_net(net):
    for node in net.nodes():
        val = int(round(random.random()))
        net.node[node]["state"] = val
    return net

def get_net_states(net):
    states = np.zeros(net.number_of_nodes())
    for node in net.nodes():
        states[node] = net.node[node]["state"]
    return states

def get_net_weights(net):
    edges = {}
    for h,t in net.edges():
        edges[(h,t)] = net[h][t]["weight"]
    return edges

def sa_burn(net, excluded_set=None, num_iters=None):
    nodes = net.nodes()
    if not num_iters:
        num_iters = net.number_of_edges() * 10 #hope this works
    for x in xrange(num_iters):
        curr_node = random.choice(nodes)
        if excluded_set:
            while curr_node in excluded_set:
                curr_node = random.choice(nodes)
        curr_input = 0
        for neighbor in net.neighbors(curr_node):
            curr_input += net.node[neighbor]["state"] * net[curr_node][neighbor]["weight"]
        if random.random() < sigmoid(curr_input):
            net.node[curr_node]["state"] = 1
        else:
            net.node[curr_node]["state"] = 0

def sa_clamp_burn(net, data):
    data = np.array(data)
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:data.shape[0]]
    for idx, seed in enumerate(top):
        net.node[seed[0]]["state"] = data[idx]
    excluded_set = set(map(operator.itemgetter(0), top))
    sa_burn(net, excluded_set)

def sa_sample(net, data=None, num_iters=None):
    total_states = np.zeros(net.number_of_nodes())
    if not num_iters:
        num_iters = net.number_of_nodes() * 10
    for x in xrange(num_iters):
        randomize_net(net)
        if data:
            sa_clamp_burn(net, data)
        else:
            sa_burn(net)
        total_states += get_net_states(net)
    return total_states

def sa_learn(net, data, num_iters=10, epsilon=0.0001):
    for x in xrange(num_iters):
        print "learn step x: ", x
        model_sample = sa_sample(net)
        data_sample = sa_sample(net, data)
        for h, t in net.edges_iter():
            delta = epsilon * (data_sample[h] * data_sample[t] - model_sample[h] * model_sample[t])
            net[h][t]["weight"] += delta
    return net

if __name__ == "__main__":
    net = load_krongraph()
    data = [0,0,1,1,0,0]
    sa_learn(net, data)
    print sa_sample(net, [0,0,1])
