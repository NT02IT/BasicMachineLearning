import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from Normalization.Normalization import Normalization
from utils.CSVHandler import CSVHandler

class UseSklearn:
    def __init__(self, datasetURL, train_size=0.5):
        self.model = LogisticRegression()

        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()   

        dataframe, label_encoders = Normalization.encode_dataframe(dataframe)
            
        X = dataframe.iloc[:, :-1]     # Chọn tất cả các hàng và cột trừ cột cuối cùng
        y = dataframe.iloc[:, -1]      # Chọn cột cuối cùng

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * train_size)
        self.X_train = X.iloc[:split_index]  
        self.y_train = y.iloc[:split_index]  
        self.X_test = X.iloc[split_index:]   
        self.y_test = y.iloc[split_index:]   

    def training(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, data_input):
        return self.model.predict(data_input)

    def testing(self):
        # Dự đoán trên tập testing
        y_pred = self.model.predict(self.X_test)

        # Tính độ chính xác so với kết quả thực tế
        accuracy = accuracy_score(self.y_test, y_pred) * 100
        print(f"Độ chính xác: {round(accuracy, 2)}%")    
    
    def getModelInfo(self):
        print("Mẫu training:", len(self.y_train))
        print("Intercept (hệ số tự do):", self.model.intercept_[0])
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_[0])