import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.CSVHandler import CSVHandler

class NaiveBayesNoLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\\naive-bayes.csv')
        dataframe = csv_handler.read_csv()
        self.X = dataframe[['Size','Number','Thickness']]
        self.y = dataframe['Lung Cancer']

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * 0.5)
        self.Xtrain = self.X.iloc[:split_index]  
        self.ytrain = self.y.iloc[:split_index]  
        self.Xtest = self.X.iloc[split_index:]   
        self.ytest = self.y.iloc[split_index:]       

    def training(self):
        print('Training...')

    def trainingFullDataset(self):
        print('Training Full Dataset...')

    def testing(self):
        print('Testing...')        

    def predict(self):
        print('Predict...')
    
    def getModelInfo(self):
        print('Model Info...')
