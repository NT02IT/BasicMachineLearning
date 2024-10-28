import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utils.CSVHandler import CSVHandler

class NaiveBayesNoLib:
    def __init__(self):
        self.model = GaussianNB()

        csv_handler = CSVHandler('Dataset\\naive-bayes.csv')
        dataframe = csv_handler.read_csv()

        # ENCODE DATA Chuyển từ chữ sang số
        columns = ['Size','Number','Thickness','Lung Cancer']
        self.label_encoders = {}  # Tạo từ điển lưu LabelEncoder cho mỗi cột
        for col in columns:
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
            self.label_encoders[col] = le  # Lưu LabelEncoder để chuyển đổi ngược lại

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
        path='Dataset\predict\\naive-bayes-nolib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()

        # ENCODE DATA Chuyển từ chữ sang số
        label_encoder = LabelEncoder()
        columns = ['Size','Number','Thickness','Lung Cancer']
        for col in columns:
            predictDataframe[col] = label_encoder.fit_transform(predictDataframe[col])

        X = predictDataframe[['Size','Number','Thickness']]
    
    def getModelInfo(self):
        print('Model Info...')
