import sklearn
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def getData(file):
    data = pd.read_excel(file)
    return data

def processData(data):
    #Get labels then delete the column
    LE = preprocessing.LabelEncoder()
    labels = data['Class']
    labels = LE.fit_transform(labels)
    data = data.drop("Class", axis=1)
    xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size=.2)

    print("xTrain: ", xTrain.shape)
    print("xTest: ", xTest.shape)
    print("yTrain: ", yTrain.shape)
    print("yTest: ", yTest.shape)

    return xTrain, xTest, yTrain, yTest

def main():
    data = getData("Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx")
    print("Shape:", data.shape)
    xTrain, xTest, yTrain, yTest = processData(data)

    #using the MLPClassifier
    mlpClass = MLPClassifier(hidden_layer_sizes=(25,15,25,5,5))
    mlpClass.fit(xTrain, yTrain)

    mlpClass.score(xTest, yTest)

    Prediction = mlpClass.predict(xTest)


    print("Accuracy: {:.2f}".format(accuracy_score(yTest, Prediction)))
    print(classification_report(yTest, Prediction, zero_division=1))


if __name__ == "__main__":
    main()