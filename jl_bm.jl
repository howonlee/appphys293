using Graphs

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

function get_tops(net, num)
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

function bm_burn(net, excluded_set=None, num_iters=NOne)
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
net = something
top = map(op.itemgetter(0), sorted(nx.degree(net).items(), key=op.itemgetter(1), reverse=True))
bm_learn(net, data)
sampled = list(sa_sample(net, completion[:392]))
sampled = [sampled[i] for i in top]
genned = np.array(sampled[0:784])
genned = genned.reshape(28, 28)
plt.imshow(genned)
plt.savefig("genned")
