import sklearn
import numpy as np
import pandas as pd
from sklearn import preprocessing
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
    xTrain, xTest, yTrain, yTest = train_test_split(data, labels)

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
    mlpClass = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 4, 5, 5, 2), random_state=1, verbose=True)
    mlpClass.fit(xTrain, yTrain)

    

if __name__ == "__main__":
    main()