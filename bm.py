import numpy as np
import numpy.random as npr
import networkx as nx
import operator
import itertools
import collections
import random

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

def sample_net(net):
    #get a random node from the net
    #list its state
    #list all the neighbors' states
    #list all the weights to the neighbors
    #then, you can do this for many nets
    pass

def flip():
    if random.random() > 0.5:
        return 1
    return -1

def node_state(val):
    #hopfield update, kind of (not really)
    if val > 0:
        return 1
    return -1


def pbm_search(net, seeds, r):
    #r is a float, omae
    marks = collections.defaultdict(int)
    vals = collections.defaultdict(int)
    nodes = dict(net.nodes(data=True))
    impossible = {} # add the seeds here?
    unused = seeds[:]
    used = []
    t1 = 0
    while unused:
        t1 += 1
        if t1 % 1000 == 0:
            print "t1: ", t1
        t2 = 0
        curr_node = unused.pop(random.randint(0, len(unused)-1))
        curr_state = nodes[curr_node]["state"]
        for neighbor in net.neighbors(curr_node):
            if impossible.has_key(neighbor):
                continue
            marks[neighbor] += 1
            vals[neighbor] += curr_state * net[curr_node][neighbor]["weight"] #times weight here eventually
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

def pbm_learn(net_data, net_model, epsilon):
    for data_edge in net_data.edges_iter():
        delta = epsilon * (data_edge[0] * data_edge[1] - model_edge[0] * model_edge[1])
        net_data.weights[data_edge] -= delta
    #and this is it
    return net_data

def running_sum(net):
    running_pos = 0
    running_neg = 0
    for first, second in net.nodes(data=True):
        if second["state"] > 0:
            running_pos += 1
        else:
            running_neg += 1

if __name__ == "__main__":
    #an old fun trick:
    #words are nodes
    #bigrams are edges
    #bam! complex network
    net = nx.Graph()
    with open("corpus.txt", "r") as corpus_file:
        words = corpus_file.read().split()
        washed, word_dict = wash(words)
        for first, second in zip(washed, washed[1:]):
            net.add_edge(first, second, weight=npr.random())
    for node, node_data in net.nodes_iter(data=True):
        node_data["state"] = flip()
    seeds = get_seeds(net, 784)
    #clamp the net values, I suppose here? I forget how to
    pgm_res = pgm_search(net, seeds, 0.75)
