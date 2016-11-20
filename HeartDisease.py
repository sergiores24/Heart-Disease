import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def featureNormalize(X):

    [m,n] = X.shape

    mean = X.mean(0)
    std = np.amax(X, axis= 0) - np.amin(X, axis= 0)

    X = (X-mean)/std

    return [X,mean,std]

def testNormalize(Xt,mean,sigma):
    Xt = (Xt-mean)/sigma
    return (Xt)

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

    [training_ds,mean,sigma]=featureNormalize(training_ds) #Normalizar datos del conjunto de entrenamiento
    validation_ds = testNormalize(validation_ds,mean,sigma)
    test_ds = testNormalize(test_ds,mean,sigma)

    logreg = LogisticRegression(C=1e5)
    logreg.fit(X=training_ds,y=np.ravel(y_training))
    predict = logreg.predict(validation_ds)


main()