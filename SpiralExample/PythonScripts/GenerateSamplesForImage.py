import numpy as np

size = 100  # element grid dimension parameter
radius = 7  # display grid dimension paramenter
xc = np.linspace(-radius, radius, num=size)
yc = np.linspace(-radius, radius, num=size)
yc = np.flip(yc)
with open("../InputFiles/Spiral_evaluation.exa", 'w') as f:
    f.write('n_inp 2\nn_out 1\nn_patt %d\n' % (size*size))
    for y in yc:
        for x in xc:
            f.write('pattern %.5f %.5f 0\n' % (x, y))
