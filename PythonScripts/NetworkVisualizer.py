import copy
from math import floor
from tkinter import *
import tkinter as tk
from tkinter.ttk import *

DIAMETER = 20
V_DIST = 10
H_DIST = 150
LINE_WIDTH = 1
BACKGROUND = 'pale green'
BUTTON_H = 30
TEXT_H = 20
f = open("test.txt", "r")


class Shape:
    def __init__(self, master=None):
        self.lines = []
        self.weights = []
        self.w_changes = []
        self.max_height = 0
        self.max_width = 0
        neurons_str = f.readline().split(" ")
        neurons_int = []
        for n in neurons_str:
            neurons_int.append(int(n))
        self.neurons = neurons_int
        self.max_weight = int(f.readline())
        self.canvas = None
        self.master = master

        f.readline()

        # Calls create method of class Shape
        self.create()

    def create(self):
        # Creates an object of class canvas
        # with the help of this we can create different shapes
        self.canvas = Canvas(self.master, background=BACKGROUND)

        for j in range(0, len(self.neurons)):
            for i in range(0, self.neurons[j]):
                self.canvas.create_oval(10 + ((DIAMETER + H_DIST) * j), 10 + ((DIAMETER + V_DIST) * i),
                                        10 + DIAMETER + ((DIAMETER + H_DIST) * j),
                                        10 + DIAMETER + ((DIAMETER + V_DIST) * i),
                                        outline="black", fill="snow", width=1)
                if 10 + DIAMETER + ((DIAMETER + V_DIST) * i) > self.max_height:
                    self.max_height = 10 + DIAMETER + ((DIAMETER + V_DIST) * i)

        self.max_width += 20 + DIAMETER + ((DIAMETER + H_DIST) * 3)
        self.max_height += 10

        # Pack the canvas to the main window and make it expandable
        self.canvas.pack(fill=BOTH, expand=1)

    def displayWeight(self, layer, s_neuron, e_neuron, value):
        if value == 0 or value > self.max_weight or value < -self.max_weight:
            return
        if value < 0:
            value *= -1
        value = int(100 - (value * 100 / self.max_weight))
        color = 'gray' + str(value)
        self.lines.append(
            self.canvas.create_line(10 + DIAMETER + (DIAMETER + H_DIST) * layer,
                                    10 + (DIAMETER / 2) + (DIAMETER + V_DIST) * s_neuron,
                                    10 + (DIAMETER + H_DIST) * (layer + 1),
                                    10 + (DIAMETER / 2) + (DIAMETER + V_DIST) * e_neuron,
                                    fill=color, width=LINE_WIDTH)
        )

    def coordinateConverter(self, weight_str):
        weight_n = 0
        weights_per_layer = []
        for l1 in range(1, len(self.neurons)):
            weight_n += self.neurons[l1 - 1] * self.neurons[l1]
            weights_per_layer.append(weight_n)

        weight = weight_str.split(" ")
        coordinate = int(weight[0])
        layer = 0
        for l2 in range(0, len(weights_per_layer)):
            if coordinate < weights_per_layer[l2]:
                layer = l2
                break
        if layer > 0:
            coordinate -= weights_per_layer[layer - 1]
        start_n = floor(coordinate / self.neurons[layer + 1])
        end_n = coordinate % self.neurons[layer + 1]
        return [layer, start_n, end_n, int(weight[1])]

    def networkInit(self):
        weight_n = 0
        weights_per_layer = []
        for l1 in range(1, len(self.neurons)):
            weight_n += self.neurons[l1 - 1] * self.neurons[l1]
            weights_per_layer.append(weight_n)
        for w in range(0, weight_n):
            weight = f.readline()
            coordinate = self.coordinateConverter(weight)
            self.weights.append(coordinate)

        self.displayNetwork(self.weights)

        f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            coordinate = self.coordinateConverter(line)
            coordinate.append(int(line.split(" ")[0]))
            self.w_changes.append(coordinate)

    def displayNetwork(self, weights):
        for line in self.lines:
            self.canvas.delete(line)
        self.lines = []
        for w in weights:
            self.displayWeight(w[0], w[1], w[2], w[3])

    def getNetworkStatus(self, iteration):
        n_weights = copy.deepcopy(self.weights)
        for i in range(0, iteration):
            w = self.w_changes[i][4]
            n_weights[w][0] = self.w_changes[i][0]
            n_weights[w][1] = self.w_changes[i][1]
            n_weights[w][2] = self.w_changes[i][2]
            n_weights[w][3] = self.w_changes[i][3]
        return n_weights


def onButtonPress(event = None):
    iterations = txt.get("end-1c linestart", "end-1c lineend")
    txt.delete(1.0, END)
    shape.displayNetwork(shape.getNetworkStatus(int(iterations)))


if __name__ == "__main__":
    # object of class Tk, responsible for creating
    # a tkinter toplevel window
    root = Tk()
    button = tk.Button(root, text='Visualize for selected iterations', height=1, command=onButtonPress)
    button.pack(side='bottom')

    txt = tk.Text(root, height=1)
    txt.pack(side='bottom')
    txt.bind('<Return>', onButtonPress)

    shape = Shape(root)

    # Sets the title to Shapes
    root.title("Network viewer")

    # Sets the geometry and position
    # of window on the screen
    dim_string = str(shape.max_width) + "x" + str(shape.max_height + BUTTON_H + TEXT_H)
    root.geometry(dim_string)

    shape.networkInit()

    print(f.readline())
    # Infinite loop breaks only by interrupt
    mainloop()
