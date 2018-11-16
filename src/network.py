import numpy as np
import pickle


@np.vectorize
def deriv(x):
    return 1. if x > 0 else 0.


class Network:

    def __init__(self, *args):
        np.random.seed(1)

        all_layers = args
        self.hidden_layers = []
        self.biases = []

        for l in zip(all_layers, all_layers[1:]):
            self.hidden_layers.append(np.random.rand(*l) * 2 - 1)
            self.biases.append(np.random.rand(l[1]) * 2 - 1)

        self.iteration = 0
        self.epoch = 0
        self.learning_rate = 0.1

    def predict(self, input_data):
        values = np.array(input_data)
        for layer, bias in zip(self.hidden_layers, self.biases):
            values = np.maximum(np.dot(values, layer) + bias, 0)
        return values

    def train(self, input_data, target):
        self.iteration += 1
        target = np.array(target)
        prediction = self.predict(input_data)
        for layer in self.hidden_layers[::-1]:
            errors = target - prediction
            gradients = deriv(prediction)
            gradients *= errors
            gradients *= self.learning_rate
            delta = errors * gradients
            print(target, prediction, errors, gradients, layer, delta)
            target = layer
            layer -= delta
            prediction = layer

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
