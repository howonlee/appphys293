using Graphs

function make_wordvec()
  function wordvec_dict(vocab_filename, vec_filename)
    ##################3
  end

  function nearest_neighbor(query_vec, data)
    ############ k only needs to be 1 for now, but k later
  end

  function load_corpus(corpus_filename)
    ##################3
  end

  function make_winows(corpus, vocab_dict)
    ############3 windows like cs224n final project
  end
  ################33
end

function make_mnist_sample()
  function get_digit(data_list, label_list, digit)
    ##############
  end

  function unpack_set(data_set)
    ##############
  end

  function unpickle_mnist(filename)
    ##############
  end
    ##############
end

function load_net_file(filename)
    ##############
end

function save_weights(net, filename)
    ##############
end

function load_weights(net, filename)
    ##############
end

function get_tops(net, num=None)
    ##############
end

function energy(net)
    ##############
end

function sigmoid(x)
    ##############
end

function randomize_net(net)
    ##############
end

function get_net_states(net):
    ##############
end

function get_net_weights(net)
    ##############
end

function bm_burn(net, excluded_set=None, num_iters=None)
    ##############
end

function bm_clamp_burn(net, data)
    ##############
end

function bm_sample(net, data=None, num_iters=None)
    ##############
end

function bm_learn(net, data)
    ##############
end

data = [1, 0, 1, 1, 0, 1]
############## all copied over from the python shit
net = load_net_file("kron2.edgelist")
tops = get_tops(net)
bm_learn(net, data)
#sampled = list(sa_sample(net, completion[:392]))
#sampled = [sampled[i] for i in top]
#genned = np.array(sampled[0:784])
#genned = genned.reshape(28, 28)
#plt.imshow(genned)
#plt.savefig("genned")
