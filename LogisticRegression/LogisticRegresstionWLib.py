import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class LogisticRegressionWLib:
    def __init__(self, dataframe):
        self.dataframe = dataframe   

    def training(self):
        X = self.dataframe[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']]
        y = self.dataframe['5-year survival']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def predict(self, X):
        input_array = pd.DataFrame([X], columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        return self.model.predict(input_array)
    
    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.model.intercept_)
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_)