import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.CSVHandler import CSVHandler
import os

class LinearRegressionWLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\linear-regression.csv')
        self.dataframe = csv_handler.read_csv()      

    def training(self):
        X = self.dataframe[['Temperature', 'Tourists', 'SunnyDays']] 
        y = self.dataframe['PredictedSales']
        self.model = LinearRegression()
        self.model.fit(X, y)     # Training      

    def predict(self):      
        path='Dataset\predict\linear-regression-withlib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()
        X = predictDataframe[['Temperature', 'Tourists', 'SunnyDays']] 
        y_hat = self.model.predict(X)
        predictDataframe['PredictedSales'] = y_hat
        csv_handler.write_csv(predictDataframe, path)
        os.startfile(path)

    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.model.intercept_)
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_)