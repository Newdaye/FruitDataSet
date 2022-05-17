import pandas as pd
import numpy as np


def getData(file):
    data = pd.read_excel(file)
    return data


def main():
    data = getData("Date_Fruit_Datasets/Date_Fruit_Datasets.xlsx")
    print(data.shape)


if __name__ == "__main__":
    main()