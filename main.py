import os
import time
import numpy as np
import pandas as pd
from Apriori.AprioriWLib import AprioriWLib
from LinearRegression.UseSklearn import UseSklearn as LinearUseSklearn
from LinearRegression.UseGradientDescent import UseGradientDescent as LinearUseGradientDescent
from LinearRegression.UseMath import UseMath as LinearUseMath
from LogisticRegression.UseSklearn import UseSklearn as LogisticUseSklearn
from LogisticRegression.UseGradientDescent import UseGradientDescent as LogisticUseGradientDescent
from NaiveBayes.UseSklearn import UseSklearn as NaiveBayesUseSklearn
from NaiveBayes.UseNaive import UseNaive as NaiveBayesUseNaive
from Normalization.Normalization import Normalization
from utils.CSVHandler import CSVHandler

def runLinearRegression():
    print("\n=====================")
    print("= LINEAR REGRESSION =")
    print("=====================\n")
    datasetURL = 'datasets\linear-regression\linear-regression.csv'

    # Linear Regression Use Sklearn
    print("- Linear Regression Use Sklearn -")
    start_time = time.time()
    
    linearUseSklearn = LinearUseSklearn(datasetURL, 0.5)
    linearUseSklearn.train()
    end_time = time.time()
    execution_time = end_time - start_time

    linearUseSklearn.getModelInfo()
    loss = linearUseSklearn.test()
    print("MSE:", round(loss, 4))
    print(f"Thời gian: {execution_time:.6f} giây")
    print(f"Dự đoán: {linearUseSklearn.predictFor([1, 2, 3])}")

    # Linear Regression Use Math
    print("\n")
    print("- Linear Regression Use Math -")
    start_time = time.time()
    linearUseMath = LinearUseMath(datasetURL)
    linearUseMath.train()
    end_time = time.time()
    execution_time = end_time - start_time

    linearUseMath.getModelInfo()
    loss = linearUseMath.test()    
    formatted_number = "{:.4e}".format(loss)
    print("MSE:", formatted_number)        
    print(f"Thời gian: {execution_time:.6f} giây")
    print(f"Dự đoán: {linearUseMath.predictFor([1, 2, 3])}")

    # Linear Regression Use Gradient Descent
    print("\n")
    print("- Linear Regression Use Gradient Descent -")
    start_time = time.time()
    
    linearUseGradientDescent = LinearUseGradientDescent(datasetURL, 0.5)
    loss_values = linearUseGradientDescent.train()
    end_time = time.time()
    execution_time = end_time - start_time

    # # Loss Visualizer
    # def plot_loss_in_thread(loss_values):
    #     linearUseGradientDescent.plot_loss(loss_values)
    # plot_thread = threading.Thread(target=plot_loss_in_thread, args=(loss_values,))
    # plot_thread.start()
    
    linearUseGradientDescent.getModelInfo()
    loss = linearUseGradientDescent.test()    

    formatted_number = "{:.4e}".format(loss)
    print("MSE:", formatted_number)        
    print(f"Thời gian: {execution_time:.6f} giây")
    print(f"Dự đoán: {linearUseGradientDescent.predictFor([1, 2, 3])}")    
    linearUseGradientDescent.plot_loss(loss_values)


def runLogisticRegression():
    print("\n=======================")
    print("= LOGISTIC REGRESSION =")
    print("=======================\n")
    datasetURL = 'datasets\logistic-regression\logistic-regression.csv'

    # Logistic Regression Use Sklearn
    print("- Logistic Regression Use Sklearn -")
    start_time = time.time()
    logisticNoLib = LogisticUseSklearn(datasetURL)
    logisticNoLib.training()
    end_time = time.time()
    execution_time = end_time - start_time

    logisticNoLib.testing()    
    print(f"Thời gian chạy: {execution_time:.6f} giây")
    logisticNoLib.getModelInfo()

    # Logistic Regression Use Gradient Descent
    print("\n")
    print("- Logistic Regression Use Gradient Descent -")
    start_time = time.time()
    logisticUseGradientDescent = LogisticUseGradientDescent(datasetURL)
    logisticUseGradientDescent.training()
    end_time = time.time()
    execution_time = end_time - start_time

    logisticUseGradientDescent.testing()    
    print(f"Thời gian chạy: {execution_time:.6f} giây")
    logisticUseGradientDescent.getModelInfo()

def runNaiveBayes():
    print("\n===============")
    print("= NAIVE BAYES =")
    print("===============\n")
    datasetURL = 'datasets\\naive-bayes\\naive-bayes.csv'

    # Naive Bayes Use Sklearn
    print("- Naive Bayes Use Sklearn -")
    start_time = time.time()
    naiveBayesUseSklearn = NaiveBayesUseSklearn(datasetURL)
    naiveBayesUseSklearn.training()
    end_time = time.time()
    execution_time = end_time - start_time
    naiveBayesUseSklearn.testing()    
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    # Naive Bayes Use Sklearn
    print("\n")
    print("- Naive Bayes Use Naive -")
    start_time = time.time()
    naiveBayesUseNaive = NaiveBayesUseNaive(datasetURL)
    naiveBayesUseNaive.training()
    end_time = time.time()
    execution_time = end_time - start_time
    naiveBayesUseNaive.testing()    
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

    # print("- Apriori No Library")
    # start_time = time.time()
    # aprioriNoLib = AprioriNoLib()
    # aprioriNoLib.training()
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Thời gian chạy: {execution_time:.6f} giây")

def runNormalization():
    print("=================")
    print("= NORMALIZATION =")
    print("=================")
    datasetURL = 'datasets\min-max-normalization\min-max-normalization.csv'
    csv_handler = CSVHandler(datasetURL)
    df = csv_handler.read_csv()

    # Chuyển đổi các dấu '?' thành NaN để dễ xử lý
    df.replace('?', pd.NA, inplace=True)
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Chuyển đổi các cột có số liệu thành kiểu số và xử lý missing data
            df[col] = df[col].apply(pd.to_numeric, errors='coerce')
            # Điền giá trị trung bình của mỗi cột cho các ô thiếu dữ liệu
            df[col] = df[col].fillna(df[col].mean())    

    print("- Min-Max Normalization With Library")
    start_time = time.time()
    df_normalized_wlib = Normalization.minMaxNormalizationWlib(df, 0, 100)    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    # Thực hiện xóa các giá trị trùng lặp trong mỗi cột
    for col in df_normalized_wlib.columns:
        df_normalized_wlib[col] = df_normalized_wlib[col].where(~df_normalized_wlib[col].duplicated(), np.nan)

    # Dịch chuyển các dữ liệu khác NAN lên đầu cột
    for col in df_normalized_wlib.columns:
        non_na = df_normalized_wlib[col].dropna()
        na_values = df_normalized_wlib[col][df_normalized_wlib[col].isna()]
        df_normalized_wlib[col] = pd.concat([non_na, na_values], axis=0, ignore_index=True)

    path='datasets\min-max-normalization\\after-normalization-wlib.csv'
    csvHandler = CSVHandler(path)
    csvHandler.write_csv(df_normalized_wlib, path)
    os.startfile(path)

    print("- Min-Max Normalization No Library")
    start_time = time.time()
    df_normalized_nolib = Normalization.minMaxNormalizationNolib(df, 0, 100)    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Thời gian chạy: {execution_time:.6f} giây")

    # Thực hiện xóa các giá trị trùng lặp trong mỗi cột
    for col in df_normalized_nolib.columns:
        df_normalized_nolib[col] = df_normalized_nolib[col].where(~df_normalized_nolib[col].duplicated(), np.nan)

    # Dịch chuyển các dữ liệu khác NAN lên đầu cột
    for col in df_normalized_nolib.columns:
        non_na = df_normalized_nolib[col].dropna()
        na_values = df_normalized_nolib[col][df_normalized_nolib[col].isna()]
        df_normalized_nolib[col] = pd.concat([non_na, na_values], axis=0, ignore_index=True)

    path='datasets\min-max-normalization\\after-normalization-nolib.csv'
    csvHandler = CSVHandler(path)
    csvHandler.write_csv(df_normalized_nolib, path)
    os.startfile(path)


def main():
    os.system('cls')
    # runLinearRegression()
    runLogisticRegression()
    # runNaiveBayes()
    # runApriori()
    # runNormalization()

if __name__ == "__main__":
    main()