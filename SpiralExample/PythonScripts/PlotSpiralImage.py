import numpy as np
from matplotlib import pyplot as plt
from spiralGridSamples import size

plt.rcParams["figure.figsize"] = [7.00, 6.00]
plt.rcParams["figure.autolayout"] = True
data2D = np.zeros([size, size])
with open("../Results/eval.exa", 'r') as f:
    f.readline()
    f.readline()
    f.readline()
    for x in range(0, size):
        for y in range(0, size):
            line = f.readline()
            if not line:
                break
            if line == "reset":
                continue
            line = line.strip().split()
            data2D[x][y] = float(line[3])

im = plt.imshow(data2D, cmap="binary")
plt.colorbar(im)
plt.savefig("../Results/eval.jpg")
plt.show()
