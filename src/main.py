from network import Network
from threelayernetwork import ThreeLayerNetwork

nn = ThreeLayerNetwork(2, 4, 2)

learning_cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
targets = [[1, 0], [0, 1], [0, 1], [1, 0]]
training_iterations = 8000


for _ in range(training_iterations):
    for case, target in zip(learning_cases, targets):
        nn.train(case, target)

for case in learning_cases:
    print(case, nn.predict(case))
