import numpy as np
import numpy.random as npr
import networkx as nx
import operator
import itertools
import collections
import random
import time
import math
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

def pbm_clamp(net, data):
    #data must be a 1d numpy array
    top = sorted(nx.degree(net).items(), key=operator.itemgetter(1), reverse=True)[:data.shape[0]]
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
    impossible = dict(zip(seeds, itertools.repeat(True)))
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

def pbm_learn(net_d, net_m, epsilon=0.1):
    #d = data, m = model
    #they better have the same topology
    states_d = nx.get_node_attributes(net_d, "state")
    states_m = nx.get_node_attributes(net_m, "state")
    degree_dict = nx.degree(net_d)
    total_delta = 0
    for data_edge in net_d.edges_iter():
        h, t = data_edge #head, tail of the edge
        #deg = 1.0 / (float(max(degree_dict[h], degree_dict[t])) ** 4.5)
        delta = epsilon * (states_d[h] * states_d[t] - states_m[h] * states_m[t]) #the network values for this
        total_delta += abs(delta)
        net_d[h][t]["weight"] += delta
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

def redo_arr(arr):
    ravelled = arr.flat
    for x in xrange(arr.size):
        if ravelled[x] == 0:
            ravelled[x] = -1
    return arr

def create_word_graph(filename="corpus.txt"):
    net = nx.Graph()
    with open(filename, "r") as corpus_file:
        words = corpus_file.read().split()
        washed, word_dict = wash(words)
        for first, second in zip(washed, washed[1:]):
            net.add_edge(first, second, weight=npr.normal() * 0.001)
    for node, node_data in net.nodes_iter(data=True):
        node_data["state"] = flip()
    return net

def create_er_graph(n=1000, p=0.050):
    #you also need the weights, which is why this is not a simple call to nx
    net = nx.fast_gnp_random_graph(n,p)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 1
    for node in net.nodes_iter():
        net.node[node]["state"] = flip()
    return net

def create_complete_graph(n=10):
    net = nx.complete_graph(n)
    for first, second in net.edges_iter():
        net[first][second]["weight"] = 0.5
    for node in net.nodes_iter():
        net.node[node]["state"] = flip()
    return net

def print_net(net):
    print "states:"
    for node in net.nodes_iter():
        print "n: %d, state: %d" % (node, net.node[node]["state"])
    print "edges: "
    for first, second in net.edges_iter():
        print "%d %d: %f" % (first, second, net[first][second]["weight"])

def net_array(net):
    net_arr = []
    for node in net.nodes_iter():
        net_arr.append(net.node[node]["state"])
    return np.array(net_arr)

def learn_step(net, data):
    """
    Data should be 1d numpy array
    """
    data = np.array(data)
    seeds = get_seeds(net, data.shape[0]) #seed INDICES
    rands = redo_arr(np.rint(npr.random(len(net))))
    pbm_model, pbm_data = pbm_clamp(net.copy(), rands), pbm_clamp(net.copy(), data)
    pbm_search(pbm_model, seeds, 0.75) #mutates
    pbm_search(pbm_data, seeds, 0.75) #mutates
    print "after searching, before learning:"
    print "energy model: ", energy(pbm_model)
    print "energy data: ", energy(pbm_data)
    return pbm_learn(pbm_data, pbm_model)

def completion_task(net, data_head, len_data):
    """
    @param net the network that encodes the BM
    @param data_head: 1d np array: what you got of the data, needs to be completed
    @param len_data: int:  the length of the data, so len(data_head) + length of the bit you need to compute
    @returns the whole 1d data array, completed
    """
    data_head = np.array(data_head)
    #seeds = get_seeds(net, len_data)
    seeds = get_seeds(net, data_head.shape[0])
    len_tail = len_data - data_head.shape[0]
    #genned_tail = np.ones(len_tail)
    genned_tail = redo_arr(np.rint(npr.random(len_tail)))
    total_data = np.hstack((data_head, genned_tail))
    net2 = pbm_clamp(net.copy(), total_data)
    pbm_search(net2, seeds, 0.75)
    return np.array(get_tops(net2, len_data))

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

def mnist_test():
    sample, completion = make_mnist_sample()
    net = create_word_graph()
    for x in xrange(1):
        print "x: ", x
        net = learn_step(net, redo_arr(np.rint(sample[x])))
    print "============"
    completion = redo_arr(np.rint(completion[:392]))
    res2 = completion_task(net, completion, 784)
    res2 = res2.reshape(28, 28)
    print res2
    plt.imshow(res2)
    plt.savefig("res2")

def small_vec_test(net):
    data = [-1, 1, -1, 1, 1, -1, -1, 1] * 98
    print np.array(get_tops(net, 784))
    print "============"
    for x in xrange(100):
        print "x: ", x
        net = learn_step(net, data)
        print "after searching, before learning:"
        print "energy model: ", energy(net)
    data2 = [-1, 1, -1, 1, 1, -1, -1, 1] * 49
    res2 = completion_task(net, data2, 784)
    res2 = res2.reshape(98, 8)
    print res2
    plt.imshow(res2)
    plt.savefig("res2_small")

def energy_test(net):
    data = np.array([-1, 1, -1, 1, 1, -1, -1, 1] * 98)
    rands = redo_arr(np.rint(npr.random(len(net))))
    seeds = get_seeds(net, data.shape[0]) #seed INDICES
    pbm_model, pbm_data = pbm_clamp(net.copy(), rands), pbm_clamp(net.copy(), data)
    orig_state = np.array(get_tops(pbm_data, len(pbm_data)))
    print "before searching, before learning:"
    print "energy model: ", energy(pbm_model)
    print "energy data: ", energy(pbm_data)
    pbm_search(pbm_model, seeds, 0.75)
    pbm_search(pbm_data, seeds, 0.75)
    print "after searching, before learning:"
    print "energy model: ", energy(pbm_model)
    print "energy data: ", energy(pbm_data)
    pbm_learn(pbm_data, pbm_model)
    pbm_model, pbm_data = pbm_clamp(pbm_data.copy(), rands), pbm_clamp(pbm_data.copy(), orig_state)
    print "before searching, after learning:"
    print "energy model: ", energy(pbm_model)
    print "energy data: ", energy(pbm_data)
    pbm_search(pbm_model, seeds, 0.75)
    pbm_search(pbm_data, seeds, 0.75)
    print "after searching, after learning:"
    print "energy model: ", energy(pbm_model)
    print "energy data: ", energy(pbm_data)
    #conclusion: great fit, shit generalization
    #heeeey, overfitting

def sample_step(net, data):
    data = np.array(data)
    seeds = get_seeds(net, data.shape[0]) #seed INDICES
    pbm_data = pbm_clamp(net.copy(), data)
    pbm_search(pbm_data, seeds, 0.75) #mutates
    return pbm_data

def model_step(net, data):
    data = np.array(data)
    seeds = get_seeds(net, data.shape[0]) #seed INDICES
    rands = redo_arr(np.rint(npr.random(len(net))))
    pbm_model = pbm_clamp(net.copy(), rands)
    pbm_search(pbm_model, seeds, 0.75) #mutates
    return pbm_model

def learn_step2(net, total_net, model_net, epsilon=0.001):
    #d = data, m = model
    #they better have the same topology
    total_delta = 0
    for h, t in net.edges_iter():
        delta = epsilon * (total_net[h] * total_net[t] - model_net[h] * model_net[t]) #the network values for this
        total_delta += abs(delta)
        net[h][t]["weight"] += delta
    print "total delta for this learn step: ", total_delta
    return net

def tiny_vec_test(net):
    data = [-1,-1,1,1,-1,-1]
    total_net = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    model_net = np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    for y in xrange(300):
        total_net += net_array(sample_step(net, data))
    print total_net
    for y in xrange(300):
        model_net += net_array(model_step(net, data))
    print model_net
    learn_step2(net, total_net, model_net)
    total_reses = np.array([0,0,0,0,0,0])
    data2 = [-1, -1, 1]
    for y in xrange(1000):
        res2 = completion_task(net, data2, 6)
        total_reses += res2
    print total_reses

if __name__ == "__main__":
    net = create_complete_graph(12)
    tiny_vec_test(net)
    #mnist_test()
    #small_vec_test(net)
    #energy_test(net)
