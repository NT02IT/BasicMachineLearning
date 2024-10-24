import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LinearRegressionWLib:
    def __init__(self, dataframe):
        self.dataframe = dataframe        

    def training(self):
        X = self.dataframe[['Temperature', 'Tourists', 'SunnyDays']] 
        y = self.dataframe['PredictedSales']
        self.model = LinearRegression()
        self.model.fit(X, y)     # Training      

    def predict(self, X):      
        input_array = pd.DataFrame([X], columns=['Temperature', 'Tourists', 'SunnyDays'])
        prediction = self.model.predict(input_array)
        return prediction[0] 

    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.model.intercept_)
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_)