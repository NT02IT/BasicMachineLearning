import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils.CSVHandler import CSVHandler
import os

class LinearRegressionWLib:
    def __init__(self):
        self.model = LinearRegression()

        csv_handler = CSVHandler('Dataset\linear-regression.csv')
        dataframe = csv_handler.read_csv()   
        self.X = dataframe[['Temperature', 'Tourists', 'SunnyDays']] 
        self.y = dataframe['PredictedSales']   

    def training(self):
        self.model.fit(self.X, self.y)  

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