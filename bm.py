import numpy as np
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
            net.add_edge(first, second)
