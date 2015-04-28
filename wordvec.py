import numpy as np
import numpy.random as npr
import random

def wordvec_dict(vocab_filename, vec_filename):
    with open(vocab_filename, "r") as vocab_file:
        vocab_list = [line.strip() for line in vocab_file]
    with open(vec_filename, "r") as vec_file:
        vec_list = [line.strip() for line in vec_file]
        vec_list = map(lambda x: np.fromstring(x, sep=" "), vec_list)
    total_vec = np.loadtxt(vec_filename)
    return dict(zip(vocab_list, vec_list)), vocab_list, total_vec

def nearest_neighbor(query_vec, dataset):
    #k = 1; use argsort if k > 1 eventually
    #use a kdtree later
    ndata = dataset.shape[0]
    sqd = np.sqrt(((dataset - query_vec) ** 2).sum(axis=1))
    return np.argmin(sqd) #Asymptotically not good

if __name__ == "__main__":
    rvec = npr.random(50) * 0.01
    vocab_dict, vocab_list, total_vec = wordvec_dict("./data/vocab.txt", "./data/wordVectors.txt")
    neigh = nearest_neighbor(rvec, total_vec)
    word = vocab_list[neigh]
    spurious_words = [random.choice(vocab_list) for x in xrange(50)]
    spurious_vecs = [vocab_dict[word] for word in spurious_words]
    print word
    print neigh
    print rvec
    print vocab_dict[word]
    print ((rvec - vocab_dict[word]) ** 2).sum()
    for vec in spurious_vecs:
        print ((rvec - vec) ** 2).sum()
