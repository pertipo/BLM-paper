import math

sample_number = 194
spiral_range = 7
spiral_count = math.floor(sample_number / 2)

with open("../InputFiles/Spiral_input.exa", 'w') as f:
    f.write("n_inp 2\nn_out 1\nn_patt {sample_number}\n")
    for i in range(0, spiral_count):
        angle = (i * math.pi) / 16
        radius = (spiral_range * (104 - i)) / 104
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        f.write("pattern " + "{:.5f}".format(x) + " " + "{:.5f}".format(y) + " 1\n")
        f.write("pattern " + "{:.5f}".format(-x) + " " + "{:.5f}".format(-y) + " 0\n")
