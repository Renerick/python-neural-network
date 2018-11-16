import numpy as np
from network import Network
from threelayernetwork import ThreeLayerNetwork
import matplotlib.pyplot as plt
import csv

nn = ThreeLayerNetwork(1, 1, 1)

# f = open("dataset.csv", "rt")
# lines = f.readlines()
# f.close

# dataset = list(csv.reader(lines, quoting=csv.QUOTE_NONNUMERIC))
# dataset = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
dataset = [[0, 1], [1, 0]]

# training_set = dataset[::2]
# test_set = dataset[1::2]

# outs = {"Iris-setosa": [1, 0, 0], "Iris-versicolor": [0, 1, 0], "Iris-virginica": [0, 0, 1]}

training_iterations = 8000

for _ in range(training_iterations):
    for case in dataset:
        nn.train([case[0]], case[1])

for test in dataset:
    print(nn.predict([test[0]]).tolist(), test[0], test[1])

print(nn)
test = np.arange(0, 1, 0.1)
values = [nn.predict([x])[0] for x in test]
plt.plot(test, values)
plt.show()
