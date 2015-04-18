import numpy as np
import numpy.random as npr
import networkx as nx
import operator
import itertools
import collections
import random
import time

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

def get_seeds(net, num_seeds):
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:num_seeds]
    return [x[0] for x in top]

def pbm_clamp(net, data):
    #data must be a list
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:len(data)]
    nodes = dict(net.nodes(data=True))
    for idx, seed in enumerate(top):
        nodes[seed[0]]["state"] = data[idx] #mutability, I hope
    return net

def sample_net(net, num_samples=50):
    nodes = net.nodes(data=True)
    nodes = random.sample(nodes, num_samples)
    for node, data in nodes:
        print "node: %d, state: %d" % (node, data["state"])
        print "degree: %d" % (net.degree(node),)

def sample_top_net(net, num_samples=30):
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:num_samples]
    nodes = dict(net.nodes(data=True))
    for seed, _ in top:
        print nodes[seed]["state"]

def flip():
    if random.random() > 0.5:
        return 1
    return -1

def node_state(val):
    #hopfield update, kind of (not really)
    if val > 0:
        return 1
    return -1

def energy(net):
    total_energy = 0
    nodes = dict(net.nodes(data=True))
    for first, second in net.edges_iter():
        weight = net[first][second]["weight"]
        total_energy -= weight * nodes[first]["state"] * nodes[second]["state"]
    return total_energy

def pbm_search(net, seeds, r):
    marks = collections.defaultdict(int)
    vals = collections.defaultdict(int)
    nodes = dict(net.nodes(data=True))
    impossible = {} # add the seeds here?
    unused = seeds[:]
    used = []
    #t1 = 0
    while unused:
        #t1 += 1
        #if t1 % 1000 == 0:
        #    print "t1: ", t1
        t2 = 0
        curr_node = unused.pop(random.randint(0, len(unused)-1))
        curr_state = nodes[curr_node]["state"]
        for neighbor in net.neighbors(curr_node):
            if impossible.has_key(neighbor):
                continue
            marks[neighbor] += 1
            #this is not good below
            vals[neighbor] += curr_state * net[curr_node][neighbor]["weight"]
            deg = net.degree(neighbor)
            req_degree = int(deg * r) + 1
            t2 += 1
            if t2 % 25000 == 0:
                break
            if marks[neighbor] > req_degree:
                unused.append(neighbor)
                nodes[neighbor]["state"] = node_state(vals[neighbor])
                impossible[neighbor] = True
        used.append(curr_node)
        #set it here, omae
    return used

def pbm_learn(net_d, net_m, epsilon=0.01):
    #d = data, m = model
    states_d = nx.get_node_attributes(net_d, "state")
    states_m = nx.get_node_attributes(net_m, "state")
    total_delta = 0
    for data_edge in net_d.edges_iter():
        h, t = data_edge #head, tail of the edge
        delta = epsilon * (states_d[h] * states_d[t] - states_m[h] * states_m[t]) #the network values for this
        total_delta += delta
        net_d[h][t]["weight"] -= delta
    print "total delta for this learn step: ", total_delta
    return net_d


def running_sum(net):
    running_pos = 0
    running_neg = 0
    for first, second in net.nodes(data=True):
        if second["state"] > 0:
            running_pos += 1
        else:
            running_neg += 1

def create_word_graph(filename="corpus.txt"):
    net = nx.Graph()
    with open(filename, "r") as corpus_file:
        words = corpus_file.read().split()
        washed, word_dict = wash(words)
        for first, second in zip(washed, washed[1:]):
            net.add_edge(first, second, weight=npr.random())
    for node, node_data in net.nodes_iter(data=True):
        node_data["state"] = flip()
    return net

if __name__ == "__main__":
    #must now test
    net = create_word_graph()
    for x in xrange(1):
        print "x: ", x
        data = [1] * 784 #reset every time, because I've been popping
        seeds = get_seeds(net, 784)
        pbm_model, pbm_data = net.copy(), pbm_clamp(net.copy(), data)
        print "============"
        sample_top_net(pbm_data)
        print "============"
        sample_top_net(pbm_model)
        print "============"
        pbm_search(pbm_model, seeds, 0.75) #mutates
        pbm_search(pbm_data, seeds, 0.75) #mutates
        net = pbm_learn(pbm_data, pbm_model)
    #time to test this sucket
