import numpy as np
import pickle


def derivative(x):
    return x if x > 0 else 0


class Network:

    def __init__(self, ins, outs, *args):
        all_layers = (ins, *args, outs)
        self.hidden_layers = []
        self.biases = []

        for l in zip(all_layers, all_layers[1:]):
            self.hidden_layers.append(np.random.rand(*l) * 2 - 1)
            self.biases.append(np.random.rand(l[1]) * 2 - 1)

    def run(self, input_data):
        values = np.array(input_data)
        for layer, bias in zip(self.hidden_layers, self.biases):
            values = np.maximum(np.dot(values, layer), 0) + bias
        return values

    def learn(self, input_data, target):
        res = self.run(input_data)
        error = res - data
        # delta = error *
        f = open("test", 'wb')
        pickle.dump(self, f)

    @staticmethod
    def load(path="model.bin"):
        f = open(path, 'rb')
        network = pickle.load(f)
        f.close()
        return network

    def save(self, path="model.bin"):
        f = open(path, 'rb')
        pickle.dump(self, f)
        f.close()
