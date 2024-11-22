import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.CSVHandler import CSVHandler
import os

class UseSklearn:
    def __init__(self, datasetURL, train_size=0.5):
        self.model = LinearRegression()

        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()   

        # Chuẩn hóa dữ liệu từ string sang số "1,2" -> 1.2
        for col in dataframe.columns:
            if dataframe[col].dtype not in ['int64', 'float64']:
                dataframe[col] = dataframe[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
            
        X = dataframe.iloc[:, :-1]     # Chọn tất cả các hàng và cột trừ cột cuối cùng
        y = dataframe.iloc[:, -1]      # Chọn cột cuối cùng

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * train_size)
        self.X_train = X.iloc[:split_index]  
        self.y_train = y.iloc[:split_index]  
        self.X_test = X.iloc[split_index:]   
        self.y_test = y.iloc[split_index:]   

    def train(self):
        self.model.fit(self.X_train, self.y_train)  

    def predict(self, data_input):    
        y_pred = self.model.predict(data_input)
        return y_pred
    
    def predictFor(self, data_input):
        data_input = np.array(data_input)
        # Chuyển data_input thành vector hàng (1, n)
        if data_input.ndim == 1:
            data_input = data_input.reshape(1, -1)
        y_pred = self.model.predict(data_input)
        return y_pred
    
    def test(self):
        predictions = self.predict(self.X_test)
        loss = mean_squared_error(self.y_test, predictions)
        return loss

    def getModelInfo(self):
        print("Mẫu training:", len(self.y_train))
        print("Hệ số tự do (Intercept):", self.model.intercept_)
        print("Hệ số của các biến độc lập (Coefficients):", self.model.coef_)