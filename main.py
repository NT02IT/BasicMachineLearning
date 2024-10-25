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

    print("- Linear Regression No Library")
    start_time = time.time()
    linearWithLib = LinearNoLib()
    linearWithLib.training()
    linearWithLib.getModelInfo()
    linearWithLib.predict()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    print("\n- Linear Regression With Library")
    start_time = time.time()
    linearWithLib = LinearWithLib()
    linearWithLib.training()
    linearWithLib.getModelInfo()
    linearWithLib.predict()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

def runLogisticRegression():
    print("\n=======================")
    print("= LOGISTIC REGRESSION =")
    print("=======================\n")

    print("- Logistic Regression No Library")
    start_time = time.time()
    logisticNoLib = LogisticNoLib()
    # logisticNoLib.training()
    logisticNoLib.trainingFullDataset()
    logisticNoLib.getModelInfo()
    logisticNoLib.testing()
    logisticNoLib.predict()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    print("\n- Logistic Regression With Library")
    start_time = time.time()
    logisticWithLib = LogisticWithLib()
    # logisticWithLib.training()
    logisticWithLib.trainingFullDataset()
    logisticWithLib.getModelInfo()
    logisticWithLib.testing()
    logisticWithLib.predict()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

def main():
    os.system('cls')
    # runLinearRegression()
    runLogisticRegression()

if __name__ == "__main__":
    main()