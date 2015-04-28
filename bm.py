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
import wordvec #own stuff

def create_complete_graph(n=10):
    net = nx.complete_graph(n)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 0.5
    for node in net.nodes_iter():
        net.node[node]["state"] = int(round(random.random()))
    net.max_node = n
    return net

def load_corpus(filename="./data/train"):
    corpus = []
    with open(filename, "r") as corpus_file:
        for line in corpus_file:
            if len(line.strip()) > 0:
                first, second = line.split()
                corpus.append(first)
    return corpus

def safe_vocab_dict(vocab_dict, query):
    try:
        return vocab_dict[query]
    except:
        return vocab_dict["UUUNKKK"]

def make_windows(corpus, vocab_dict):
    trips = zip(corpus, corpus[1:], corpus[2:])
    tripvecs = []
    for trip in trips:
        tripvecs.append(np.hstack([safe_vocab_dict(vocab_dict, word) for word in trip]))
    return tripvecs

def load_file(filename="kron.edgelist"):
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
    train_ones = get_digit(train_data, train_labels, 1)
    random.shuffle(train_zeros) #inplace
    sample, completion = train_data[:50], train_data[1000]
    return sample, completion

def save_weights(net, filename="net_weights"):
    net_weights = np.zeros((net.max_node, net.max_node))
    for h,t in net.edges_iter():
        net_weights[h][t] = net[h][t]["weight"]
    np.save(filename, net_weights)
    print "net saved"

def load_weights(net, filename="net_weights"):
    #must have the right net beforehands
    #meaning, unpickle that crap
    net_weights = np.load(filename)
    for h,t in net.edges_iter():
        net[h][t]["weight"] = net_weights[h][t]
    return net

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
        num_iters = net.number_of_nodes() // 2  #this works, why does it work
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
        num_iters = 20 #is this kosher?
    for x in xrange(num_iters):
        #print >> sys.stderr, "sample step x: ", x, " / ", num_iters
        randomize_net(net)
        if data is not None:
            sa_clamp_burn(net, data)
        else:
            sa_burn(net)
        total_states += get_net_states(net)
    return total_states

def sa_learn(net, data, num_iters=None, epsilon=0.01):
    if not num_iters:
        num_iters = len(data)
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

def mnist_test():
    #not the test stage in testing, just actually a proof of concept
    net = load_file("kron2.edgelist")
    data, completion = make_mnist_sample()
    top = map(op.itemgetter(0), sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True))
    sa_learn(net, data)
    sampled = list(sa_sample(net, completion[:392]))
    sampled = [sampled[i] for i in top]
    genned = np.array(sampled[0:784])
    genned = genned.reshape(28, 28)
    plt.imshow(genned)
    plt.savefig("genned")

def vocab_test(net_file=None):
    net = load_file("kron2.edgelist")
    vocab_dict, vocab_list, total_vec = wordvec.wordvec_dict("./data/vocab.txt", "./data/wordVectors.txt")
    train = load_corpus()
    windows = make_windows(train, vocab_dict)
    print len(windows)
    sys.exit(0)
    windows = windows[:500]
    if not net_file:
        #no labels as of yet, just the representations
        sa_learn(net, windows)
        save_weights(net, "vocab_net2")
    else:
        net = load_weights(net, net_file)
    print "total net learned now"
    curr_window = windows[-1]
    while True:
        cut_window = curr_window[0:100]
        top = map(op.itemgetter(0), sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True))[0:150]
        sampled = list(sa_sample(net, cut_window))
        curr_window = np.array([sampled[i] for i in top])
        next_wordvec = curr_window[100:]
        print vocab_list[wordvec.nearest_neighbor(next_wordvec, total_vec)]

if __name__ == "__main__":
    #i named everything sa
    #shit ain't sa, it's just gd
    mnist_test()
