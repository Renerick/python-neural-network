from src.network import Network

nn = Network(2, 2)
nn = Network.load()

print(nn.run([1] * 2))
# nn.learn([1,1], [0])
