import numpy as np

def wordvec_dict(vocab_filename, vec_filename):
    with open(vocab_filename, "r") as vocab_file:
        vocab_list = [line.strip() for line in vocab_file]
    with open(vec_filename, "r") as vec_file:
        vec_list = [line.strip() for line in vec_file]
        vec_list = map(lambda x: np.fromstring(x, sep=" "), vec_list)
    total_vec = np.loadtxt(vec_filename)
    return dict(zip(vocab_list, vec_list)), vocab_list, total_vec

if __name__ == "__main__":
    print wordvec_dict("./data/vocab.txt", "./data/wordVectors.txt")[1]
