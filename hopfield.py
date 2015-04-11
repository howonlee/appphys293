import numpy as np

class HopNet(object):
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.uniform(-1.0, 1.0, (num_inputs, num_inputs))

    def evaluate(self, input_pattern):
        sums = input_pattern.dot(self.weights)
        s = np.zeros(self.num_inputs)
        for i, value in enumerate(sums):
            if value > 0:
                s[i] = 1
            else:
                s[i] = -1
        return s

    def run(self, input_pattern, iters=10):
        last_input = input_pattern
        iter_ct = 0
        while True:
            result = self.evaluate(last_input)
            iter_ct += 1
            if np.array_equal(result, last_input) or iter_ct == iters:
                return result
            else:
                last_input = result

    def hebbian_train(self, input_patterns):
        n = len(input_patterns)
        new_weights = np.zeros_like(self.weights)
        for i in xrange(self.num_inputs):
            for j in xrange(self.num_inputs):
                if i == j:
                    continue
                for m in xrange(n):
                    new_weights[i,j] += input_patterns[m][i] * input_patterns[m][j]
        new_weights *= 1 / float(n)
        self.weights = new_weights
