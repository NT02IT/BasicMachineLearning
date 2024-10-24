import numpy as np
from utils.CSVHandler import CSVHandler
from LinearRegression.LinearRegressionNoLib import LinearRegressionNoLib as LR0Lib
from LinearRegression.LinearRegressionWLib import LinearRegressionWLib as LRWLib

def main():
    csv_handler = CSVHandler('Dataset\linear-regression.csv')
    dataframe = csv_handler.read_csv()
    print(dataframe)

    print('--------------------------------')
    print("Linear Regression No Library")
    lr0Lib = LR0Lib(dataframe)
    lr0Lib.training()
    lr0Lib.getModelInfo()
    print(f"[[93, 1212, 6]] => y = {lr0Lib.predict([93, 1212, 6])}")

    print('--------------------------------')
    print("Linear Regression With Library")
    lrwLib = LRWLib(dataframe)
    lrwLib.training()
    lrwLib.getModelInfo()
    print(f"[[93, 1212, 6]] => y = {lrwLib.predict([93, 1212, 6])}")

if __name__ == "__main__":
    main()