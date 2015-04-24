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

#here is the suspicion:
#uncool kids' gradient descent on a hopfield net with cool kids topology
#will work way better than expected
#because of sparsity and percolation
#very strong partial recognition

if __name__ == "__main__":
    pass
