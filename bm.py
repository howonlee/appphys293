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

def wash(words):
    curr_word = 0
    washed = []
    word_dict = {}
    for word in words:
        if word in word_dict:
            washed.append(word_dict[word])
        else:
            word_dict[word] = curr_word
            curr_word += 1
    return washed, word_dict

def create_word_graph(filename="corpus.txt"):
    net = nx.Graph()
    with open(filename, "r") as corpus_file:
        words = corpus_file.read().split()
        washed, word_dict = wash(words)
        for first, second in zip(washed, washed[1:]):
            net.add_edge(first, second, weight=npr.normal() * 0.1)
    for node, node_data in net.nodes_iter(data=True):
        node_data["state"] = flip()
    return net

def create_complete_graph(n=10):
    net = nx.complete_graph(n)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 0.5
    for node in net.nodes_iter():
        net.node[node]["state"] = flip()
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

def bm_clamp(net, data):
    #data must be a 1d numpy array
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:data.shape[0]]
    nodes = dict(net.nodes(data=True))
    for idx, seed in enumerate(top):
        nodes[seed[0]]["state"] = data[idx] #mutability, I hope
    return net

def energy(net):
    total_energy = 0
    nodes = dict(net.nodes(data=True))
    for first, second in net.edges_iter():
        weight = net[first][second]["weight"]
        total_energy -= weight * nodes[first]["state"] * nodes[second]["state"]
    return total_energy

def bm_search(net, seeds, num_iters=1000):
    for x in xrange(num_iters):
        pass#################
    return net

def bm_sim(net, num_iters=1000):
    for x in xrange(num_iters):
        pass###############
    return net

def sample_step(net, data):
    data = np.array(data)
    seeds = get_seeds(net, data.shape[0]) #seed INDICES
    bm_data = bm_clamp(net.copy(), data)
    for x in xrange(10):
        bm_search(bm_data, seeds) #mutates
    return bm_data

def model_step(net, data):
    data = np.array(data)
    seeds = get_seeds(net, data.shape[0]) #seed INDICES
    rands = redo_arr(np.rint(npr.random(len(net))))
    bm_model = bm_clamp(net.copy(), rands)
    for x in xrange(10):
        bm_search(bm_model, seeds) #mutates
    return bm_model

def tiny_vec_test(net):
    data = [-1,-1,1,1,-1,-1]
    for x in xrange(50):
        total_net = np.array([0,0,0,0,0,0,0,0,0,0])
        model_net = np.array([0,0,0,0,0,0,0,0,0,0])
        for y in xrange(300):
            total_net += net_array(sample_step(net, data))
        print total_net
        for y in xrange(300):
            model_net += net_array(model_step(net, data))
        print model_net

if __name__ == "__main__":
    net = create_complete_graph()
    tiny_vec_test(net)
