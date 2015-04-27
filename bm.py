import numpy as np
import numpy.random as npr
import networkx as nx
import operator as op
import itertools
import collections
import random
import datetime
import sys
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
    net.max_node = n
    return net

def load_krongraph(filename="kron.edgelist"):
    net = nx.Graph()
    #note kron edgelist 1 indexed because they are terrible
    max_node = -1
    with open(filename) as kron_file:
        for line in kron_file:
            first, second = line.split()
            net.add_edge(int(first), int(second))
            if int(first) > max_node:
                max_node = int(first)
            if int(second) > max_node:
                max_node = int(second)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 0.5
    for node in net.nodes_iter():
        net.node[node]["state"] = int(round(random.random()))
    net.max_node = max_node + 1
    return net

def unpickle_mnist(filename="mnist.pkl.gz"):
    with gzip.open(filename, "rb") as gzip_file:
        train_set, valid_set, test_set = cPickle.load(gzip_file)
    return train_set, valid_set, test_set

def unpack_set(data_set):
    data, labels = data_set
    data_list, label_list = [], []
    for x in xrange(data.shape[0]):
        data_list.append(data[x, :])
        label_list.append(labels[x])
    return data_list, label_list

def get_digit(data_list, label_list, digit):
    return [x[1] for x in filter(lambda x: label_list[x[0]] == digit, enumerate(data_list))]

def make_mnist_sample():
    train_set, valid_set, test_set = unpickle_mnist()
    train_data, train_labels = unpack_set(train_set)
    train_zeros = get_digit(train_data, train_labels, 0)
    random.shuffle(train_zeros) #inplace
    sample, completion = train_zeros[:500], train_zeros[1000]
    return sample, completion

def get_seeds(net, num_seeds):
    top = sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True)[:num_seeds]
    return [x[0] for x in top]

def get_tops(net, num_seeds):
    top = sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True)[:num_seeds]
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
    states = np.zeros(net.max_node)
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
        num_iters = net.number_of_nodes() * 2 #hope this works
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
    top = sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True)[:data.shape[0]]
    for idx, seed in enumerate(top):
        net.node[seed[0]]["state"] = data[idx]
    excluded_set = set(map(op.itemgetter(0), top))
    sa_burn(net, excluded_set)

def sa_sample(net, data=None, num_iters=None):
    total_states = np.zeros(net.max_node)
    if not num_iters:
        num_iters = 10 #is this kosher?
    for x in xrange(num_iters):
        #print >> sys.stderr, "sample step x: ", x, " / ", num_iters
        randomize_net(net)
        if data is not None:
            sa_clamp_burn(net, data)
        else:
            sa_burn(net)
        total_states += get_net_states(net)
    return total_states

def sa_learn(net, data, num_iters=None, epsilon=0.0001):
    if not num_iters:
        #num_iters = len(data)
        num_iters = 50
        print >> sys.stderr, "total iters: ", num_iters
    for x in xrange(num_iters):
        #do you even python 2.6 bro
        print >> sys.stderr, "learn step x: ", x, datetime.datetime.now()
        curr_data = random.choice(data)
        model_sample = sa_sample(net)
        data_sample = sa_sample(net, curr_data)
        for h, t in net.edges_iter():
            delta = epsilon * (data_sample[h] * data_sample[t] - model_sample[h] * model_sample[t])
            net[h][t]["weight"] += delta
    return net

if __name__ == "__main__":
    #i named everything sa
    #shit ain't sa, it's just gd
    net = load_krongraph("kron2.edgelist")
    #net = create_complete_graph(n=2048)
    data, completion = make_mnist_sample()
    top = map(op.itemgetter(0), sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True))
    sa_learn(net, data)
    sampled = list(sa_sample(net, completion[:392]))
    sampled = [sampled[i] for i in top]
    genned = np.array(sampled[0:784])
    genned = genned.reshape(28, 28)
    plt.imshow(genned)
    plt.savefig("genned")
