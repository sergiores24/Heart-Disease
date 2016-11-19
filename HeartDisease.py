import numpy as np
import random


def main():

    f = open('Cleveland_DataSet.csv', 'r')

    np.set_printoptions(suppress=True)

    data = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        line = [float(i) for i in line]
        data.append(line)

    x = np.matrix(data)

    [m, n] = x.shape
    t = random.sample(range(m), round((m * 0.5)))
    training_ds = x[t, :]

    t2 = []
    for element in range(m):
        if element not in t:
            t2.append(element)

    x_half = x[t2, :]
    [m, n] = x_half.shape
    t = random.sample(range(m), round((m * 0.5)))
    validation_ds = x_half[t, :]
    t2 = []
    for element in range(m):
        if element not in t:
            t2.append(element)
    test_ds=x_half[t2,:]

    y_training=training_ds[:,n-1]
    y_validation=validation_ds[:,n-1]
    y_test=test_ds[:,n-1]




main()