import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils.CSVHandler import CSVHandler

class UseMath:
    def __init__(self, datasetURL, train_size=0.5):
        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()

        # Chuẩn hóa dữ liệu từ string sang số "1,2" -> 1.2
        for col in dataframe.columns:
            if dataframe[col].dtype not in ['int64', 'float64']:
                dataframe[col] = dataframe[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

        X = dataframe.iloc[:, :-1].values  # Các biến độc lập
        y = dataframe.iloc[:, -1].values   # Biến phụ thuộc

        # Thêm cột 1 cho hệ số tự do (intercept)
        X = np.c_[np.ones(X.shape[0]), X]  # Thêm cột 1 vào đầu

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * train_size)
        self.X_train = X[:split_index]  # Dữ liệu training
        self.y_train = y[:split_index]  # Dữ liệu mục tiêu training
        self.X_test = X[split_index:]   # Dữ liệu testing
        self.y_test = y[split_index:]   # Dữ liệu mục tiêu testing

    def train(self):
        # Áp dụng công thức Normal Equation để tìm hệ số
        # theta = (X.T * X)^-1 * X.T * y
        X_transpose = self.X_train.T  # Chuyển vị của X
        self.theta = np.linalg.inv(X_transpose.dot(self.X_train)).dot(X_transpose).dot(self.y_train)

    def predict(self, data_input):
        # Dự đoán y = X * theta
        y_pred = data_input.dot(self.theta)
        return y_pred
    
    def predictFor(self, data_input):
        data_input = np.array(data_input)
        # Chuyển data_input thành vector hàng (1, n)
        if data_input.ndim == 1:
            data_input = data_input.reshape(1, -1)
        # Thêm cột hệ số tự do (bias term) vào đầu mảng
        data_input = np.c_[np.ones(data_input.shape[0]), data_input]
        # Dự đoán y = X * theta
        y_pred = data_input.dot(self.theta)
        return y_pred

    def test(self):
        # Kiểm tra mô hình với dữ liệu kiểm tra
        predictions = self.predict(self.X_test)
        loss = mean_squared_error(self.y_test, predictions)
        return loss

    def getModelInfo(self):
        print("Mẫu training:", len(self.y_train))
        print("Hệ số tự do (Intercept):", self.theta[0])  # Hệ số tự do là phần tử đầu tiên của theta
        print("Hệ số của các biến độc lập (Coefficients):", self.theta[1:])  # Các hệ số còn lại là của các biến độc lập
