import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as mt
from sklearn import svm
import os

def accuracy (TP, TN, P, N):
    return ((TP+TN)/(P+N))

def sensitivity (TP,P):
    return (TP/P)
def specificity (TN,N):
    return (TN/N)
def F1Score(TP,FP,FN):
    return ((2*TP)/((2*TP)+FP+FN))

def Print(Cmatrix,name):
    print("\n--------------------------Confusion ",name," -------------------------")
    print("                    || Predicted Condition Positive || Predicted Condition Negative || \n"
          "Condition Positive  ||             ", Cmatrix[0, 0], "             ||          ", Cmatrix[0, 1],
          "                 || ",
          "\nCondition Negative  ||             ", Cmatrix[1, 0], "              ||          ", Cmatrix[1, 1],
          "                || ")
    print("\nAccuracy: ",
          accuracy(Cmatrix[0, 0], Cmatrix[1, 1], (Cmatrix[0, 0] + Cmatrix[0, 1]), (Cmatrix[1, 0] + Cmatrix[1, 1])))
    print("Sensitivity: ", sensitivity(Cmatrix[0, 0], (Cmatrix[0, 0] + Cmatrix[0, 1])))
    print("Specificity: ", specificity(Cmatrix[1, 1], (Cmatrix[1, 1] + Cmatrix[1, 0])))
    print("F1Score: ", F1Score(Cmatrix[0, 0], Cmatrix[1, 0], Cmatrix[0, 1]))

def LogisticR():
    logreg = LogisticRegression(C=5)
    logreg.fit(X=training_ds,y=np.ravel(y_training))
    score = cross_val_score(logreg, validation_ds, y=np.ravel(y_validation), cv = 5)
    predict=logreg.predict(test_ds)
    Cmatrix = mt.confusion_matrix(predict,y_test)
    Print(Cmatrix,"Logistic Regression")
    print("\nCV Accuracy LogisticRegression = ", np.mean(score) * 100)

def SupportVM():

    clf = svm.SVC(C= 1,kernel='poly',degree=3)
    clf.fit(X = training_ds, y = np.ravel(y_training))
    score = cross_val_score(clf, validation_ds, y=np.ravel(y_validation), cv = 5)
    predict = clf.predict(test_ds)
    Cmatrix = mt.confusion_matrix(predict,y_test)
    Print(Cmatrix,"Support Vector Machine")
    print("\nCV Accuracy SVM = ", np.mean(score) * 100)

def PerceptronML():
    NeuralNet=MLPClassifier(solver="adam",hidden_layer_sizes=(25,),activation="logistic",max_iter=3000)
    NeuralNet.fit(X=training_ds, y=np.ravel(y_training))
    score = cross_val_score(NeuralNet, validation_ds, y=np.ravel(y_validation), cv=5)
    predict = NeuralNet.predict(test_ds)
    Cmatrix = mt.confusion_matrix(predict, y_test)
    Print(Cmatrix,"Multi Layer Perceptron")
    print("\nCV Accuracy MultiLayer Perceptron = ", np.mean(score) * 100)



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
    LogisticR()
    SupportVM()
    PerceptronML()

main()