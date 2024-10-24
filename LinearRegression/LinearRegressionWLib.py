import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionWLib:
    def __init__(self, dataframe):
        self.dataframe = dataframe        

    def training(self):
        X = self.dataframe[['Temperature', 'Tourists', 'SunnyDays']] 
        y = self.dataframe['PredictedSales']
        self.model = LinearRegression()
        self.model.fit(X, y)     # Training      

    def predict(self, X):      
        # Chuyển đổi đầu vào thành dạng 2D
        input_array = pd.DataFrame([X], columns=['Temperature', 'Tourists', 'SunnyDays'])
        
        # Dự đoán
        prediction = self.model.predict(input_array)
        return prediction[0] 

    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.model.intercept_)
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_)