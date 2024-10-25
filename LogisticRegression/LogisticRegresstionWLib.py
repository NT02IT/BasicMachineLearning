import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from utils.CSVHandler import CSVHandler

class LogisticRegressionWLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\logistic-regression.csv')
        dataframe = csv_handler.read_csv()
        self.X = dataframe[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']]
        self.y = dataframe['5-year survival']

        # Chia đôi dữ liệu 80% dùng để training 20% dùng để testing
        split_index = int(len(dataframe) * 0.5)
        self.Xtrain = self.X.iloc[:split_index]  
        self.ytrain = self.y.iloc[:split_index]  
        self.Xtest = self.X.iloc[split_index:]   
        self.ytest = self.y.iloc[split_index:]   

    def training(self):
        self.model = LogisticRegression()
        self.model.fit(self.Xtrain, self.ytrain)

    def trainingFullDataset(self):
        self.model = LogisticRegression()
        self.model.fit(self.X, self.y)

    def testing(self):
        # Dự đoán trên tập testing
        y_pred = self.model.predict(self.Xtest)

        # Tính độ chính xác so với kết quả thực tế
        accuracy = accuracy_score(self.ytest, y_pred) * 100
        print(f"Độ chính xác: {accuracy}%")

    def predict(self):
        path='Dataset\predict\logistic-regression-withlib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()
        X = predictDataframe[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']]
        predictDataframe['5-year survival'] = self.model.predict(X)
        csv_handler.write_csv(predictDataframe, path)
        os.startfile(path)
    
    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.model.intercept_[0])
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_[0])