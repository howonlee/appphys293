import numpy as np

if __name__ == "__main__":
    #an old fun trick:
    #words are nodes
    #bigrams are edges
    #bam! complex network
    with open("corpus.txt", "r") as corpus_file:

