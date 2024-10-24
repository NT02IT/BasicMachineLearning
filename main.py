import os
import time
import numpy as np
from utils.CSVHandler import CSVHandler
from LinearRegression.LinearRegressionNoLib import LinearRegressionNoLib as LinearNoLib
from LinearRegression.LinearRegressionWLib import LinearRegressionWLib as LinearWithLib
from LogisticRegression.LogisticRegressionNoLib import LogisticRegressionNoLib as LogisticNoLib
from LogisticRegression.LogisticRegresstionWLib import LogisticRegressionWLib as LogisticWithLib

def runLinearRegression():
    print("\n=====================")
    print("= LINEAR REGRESSION =")
    print("=====================\n")
    csv_handler = CSVHandler('Dataset\linear-regression.csv')
    dataframe = csv_handler.read_csv()

    print("- Linear Regression No Library")
    start_time = time.time()
    linearWithLib = LinearNoLib(dataframe)
    linearWithLib.training()
    linearWithLib.getModelInfo()
    print(f"Predict: [[93, 1212, 6]] => y = {linearWithLib.predict([93, 1212, 6])}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    print("\n- Linear Regression With Library")
    start_time = time.time()
    linearWithLib = LinearWithLib(dataframe)
    linearWithLib.training()
    linearWithLib.getModelInfo()
    print(f"Predict: [[93, 1212, 6]] => y = {linearWithLib.predict([93, 1212, 6])}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

def runLogisticRegression():
    print("\n=======================")
    print("= LOGISTIC REGRESSION =")
    print("=======================\n")
    csv_handler = CSVHandler('Dataset\logistic-regression.csv')
    dataframe = csv_handler.read_csv()

    print("- Logistic Regression No Library")
    start_time = time.time()
    logisticNoLib = LogisticNoLib(dataframe)
    logisticNoLib.training()
    logisticNoLib.getModelInfo()
    print(f"Predict: [55.5, 97.5, 8.8, 28.2, 72.2] => y = {logisticNoLib.predict([55.5, 97.5, 8.8, 28.2, 72.2])}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    print("\n- Logistic Regression With Library")
    start_time = time.time()
    logisticWithLib = LogisticWithLib(dataframe)
    logisticWithLib.training()
    logisticWithLib.getModelInfo()
    print(f"Predict: [55.5, 97.5, 8.8, 28.2, 72.2] => y = {logisticWithLib.predict([55.5, 97.5, 8.8, 28.2, 72.2])}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

def main():
    os.system('cls')

    runLinearRegression()
    runLogisticRegression()

if __name__ == "__main__":
    main()