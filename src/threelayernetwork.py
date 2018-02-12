import numpy as np
import pickle


@np.vectorize
def relu_d(x):
    return 1. if x > 0 else 0.


def relu(x):
    return np.maximum(x, 0)


def sig(x):
    return 1 / (1 + np.exp(-x))


def sig_d(x):
    return x * (1 - x)


class ThreeLayerNetwork:
    '''
    Represent simple neural network with 3 layers
    '''
    def __init__(self, ins, hidden, outs):

        self.hl_weights = np.random.rand(hidden, ins) - 0.5
        self.ol_weights = np.random.rand(outs, hidden) - 0.5

        self.hl_biases = np.random.rand(hidden, 1) - 0.5
        self.ol_biases = np.random.rand(outs, 1) - 0.5

        self.learning_rate = 0.3

        self.act = sig
        self.der = sig_d

    def predict(self, input_data):
        values = np.array(input_data, ndmin=2).T
        values = self.act(np.dot(self.hl_weights, values) + self.hl_biases)
        values = self.act(np.dot(self.ol_weights, values) + self.ol_biases)
        return values

    def train(self, input_data, target):
        input_data = np.array(input_data, ndmin=2).T
        target = np.array(target, ndmin=2).T

        hidden_in = np.dot(self.hl_weights, input_data) + self.hl_biases
        hidden_out = self.act(hidden_in)

        out_in = np.dot(self.ol_weights, hidden_out) + self.ol_biases
        out_out = self.act(out_in)

        out_errors = target - out_out

        out_grad = out_errors * self.der(out_out)
        out_delta = np.dot(out_grad, hidden_out.T) * self.learning_rate
        # print("out_grad ", out_grad, "|out_delta ", out_delta)

        self.ol_weights += out_delta
        self.ol_biases += out_grad

        h_errors = np.dot(self.ol_weights.T, out_errors)

        h_grad = h_errors * self.der(hidden_out)
        h_delta = np.dot(h_grad, input_data.T) * self.learning_rate
        # print("h_grad ", h_grad, "|h_delta ", h_delta)

        self.hl_weights += h_delta
        self.hl_biases += h_grad
        # print(out_errors)

    @staticmethod
    def load(path="model.bin"):
        f = open(path, 'rb')
        network = pickle.load(f)
        f.close()
        return network

    def save(self, path="model.bin"):
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()

    def __repr__(self):
        return "{} {} \n{} {}".format(self.hl_weights,
                                      self.hl_biases,
                                      self.ol_weights,
                                      self.ol_biases)
