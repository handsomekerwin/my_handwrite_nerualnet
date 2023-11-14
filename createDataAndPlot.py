import numpy as np
import pandas as pd
import math
import random
from colorama import Fore, Back, Style
import matplotlib.pyplot as plt

NumOfPoints = 100

def create_data(numofpoints):
    entry_list = []  # list of lists
    for i in range(numofpoints):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        tag = tag_entry([x,y])
        entry_list.append([x,y,tag])
    return np.array(entry_list)

def tag_entry(x):
    if x[0]**2 + x[1]**2 <= 1:
        return 0
    else:
        return 1

def plot_data(data, table_name):
    color = []
    for i in data[:,2]:
        if i == 0:
            color.append('blue')
        else:
            color.append('red')
    plt.scatter(data[:,0], data[:,1], c=color)
    plt.title(table_name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == "__main__":



    # __________TEST__________
    print(Fore.YELLOW )
    data = create_data(NumOfPoints)
    print(create_data(NumOfPoints))
    plot_data(data)


