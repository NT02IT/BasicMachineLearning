import os
import time
import numpy as np
from Apriori.AprioriWLib import AprioriWLib
from MinMaxNormalization.MinMaxScalerNoLib import MinMaxScalerNoLib
from NaiveBayes.NaiveBayesNoLib import NaiveBayesNoLib
from NaiveBayes.NaiveBayesWLib import NaiveBayesWLib 
from LinearRegression.LinearRegressionNoLib import LinearRegressionNoLib as LinearNoLib
from LinearRegression.LinearRegressionWLib import LinearRegressionWLib as LinearWithLib
from LogisticRegression.LogisticRegressionNoLib import LogisticRegressionNoLib as LogisticNoLib
from LogisticRegression.LogisticRegresstionWLib import LogisticRegressionWLib as LogisticWithLib
from MinMaxNormalization.MinMaxScalerWLib import MinMaxScalerWLib 

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

def runNaiveBayes():
    print("\n===============")
    print("= NAIVE BAYES =")
    print("===============\n")

    print("- Naive Bayes No Library")
    start_time = time.time()
    naiveBayesNoLib = NaiveBayesNoLib()
    # naiveBayesNoLib.training()
    naiveBayesNoLib.trainingFullDataset()
    naiveBayesNoLib.testing()
    naiveBayesNoLib.predict()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")


    print("\n- Naive Bayes With Library")
    start_time = time.time()
    naiveBayesWLib = NaiveBayesWLib()
    # naiveBayesWLib.training()
    naiveBayesWLib.trainingFullDataset()
    naiveBayesWLib.testing()
    naiveBayesWLib.predict()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

def runApriori():
    print("\n===========")
    print("= APRIORI =")
    print("===========\n")

    print("- Apriori With Library")
    start_time = time.time()
    aprioriWLib = AprioriWLib()
    aprioriWLib.training()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

def runMinMaxNormalization():
    print("=========================")
    print("= MIN-MAX NORMALIZATION =")
    print("=========================")

    print("- Min-Max Normalization With Library")
    start_time = time.time()
    minmaxScalerWLib = MinMaxScalerWLib()
    minmaxScalerWLib.Normalization()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    print("- Min-Max Normalization No Library")
    start_time = time.time()
    minmaxScalerNoLib = MinMaxScalerNoLib()
    minmaxScalerNoLib.Normalization()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")


def main():
    os.system('cls')
    # runLinearRegression()
    # runLogisticRegression()
    # runNaiveBayes()
    # runApriori()
    runMinMaxNormalization()

if __name__ == "__main__":
    main()