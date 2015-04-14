import numpy as np
import numpy.random as npr
import networkx as nx

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

def pgm(net1, net2, seeds, r): #seeds is a list of tups
    marks = collections.defaultdict(int)
    #heap?
    imp_1 = {} #impossible tails
    imp_2 = {} #impossible heads
    unused = seeds[:]
    used = []
    #to make it online algo: unused should be of high priority, but used should also be taken fron
    while unused:
        t2 = 0
        curr_pair = unused.pop(random.randint(0,len(unused)-1))
        for neighbor in itertools.product(net1.neighbors(curr_pair[0]), net2.neighbors(curr_pair[1])):
            #take the filter out of the loops
            if imp_1.has_key(neighbor[0]) or imp_2.has_key(neighbor[1]):
                continue
            marks[neighbor] += 1
            t2 += 1
            if t2 % 250000 == 0:
                #this is an awful hack
                break
            #take it out, I guess?
            if marks[neighbor] > r:
                unused.append(neighbor)
                imp_1[neighbor[0]] = True
                imp_2[neighbor[1]] = True
        #maximum of the marks here, but later
        used.append(curr_pair)
    return used

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
        node_data["state"] = 0
