import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import sklearn.metrics as mt

def LogisticR():
    logreg = LogisticRegression(C=5)
    logreg.fit(X=training_ds,y=np.ravel(y_training))
    print(cross_val_score(logreg, validation_ds, y=np.ravel(y_validation), cv = 5))
    predict=logreg.predict(test_ds)
    print(mt.confusion_matrix(predict,y_test))

def SupportVM():
    

def main():

    global y_training, training_ds,y_validation,y_test,test_ds,validation_ds
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
    logisticr()

main()