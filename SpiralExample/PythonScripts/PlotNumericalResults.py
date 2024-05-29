import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [8.00, 8.00]
plt.rcParams["figure.autolayout"] = True

errF = open("../Results/log.txt", 'r')
genF = open("../Results/test_log.txt", 'r')

genInterval = 1000

errLines = errF.readlines()
genLines = genF.readlines()

sizeErr = len(errLines) - 2
sizeGen = len(genLines) - 3

xErr = np.zeros([sizeErr])
yErr = np.zeros([sizeErr])
xGen = np.zeros([sizeGen])
yGen = np.zeros([sizeGen])

for i in range(2, sizeErr):
    xErr[i] = i
    line = errLines[i].strip().split()
    yErr[i] = line[1]

for i in range(3, sizeGen):
    xGen[i] = i * genInterval
    line = genLines[i].strip().split()
    yGen[i] = line[1]


plt.xlim(0, sizeErr)
plt.ylim(0, 0.6)
plt.grid()
plt.plot(xErr, yErr)
plt.plot(xGen, yGen)
# plt.savefig("../Results/Iterations.png")
plt.show()

errF.close()
genF.close()
