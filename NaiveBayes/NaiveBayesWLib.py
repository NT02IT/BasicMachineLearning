import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from utils.CSVHandler import CSVHandler
from Validate.ValidateNoLib import ValidateNoLib

class NaiveBayesWLib:
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
        split_index = int(len(dataframe) * 0.8)
        self.Xtrain = self.X.iloc[:split_index]  
        self.ytrain = self.y.iloc[:split_index]  
        self.Xtest = self.X.iloc[split_index:]   
        self.ytest = self.y.iloc[split_index:]       

    def training(self):
        self.model.fit(self.Xtrain, self.ytrain)

    def trainingFullDataset(self):
        self.model.fit(self.X, self.y)

    def testing(self):
        # Dự đoán trên tập testing
        y_pred = self.model.predict(self.Xtest)

        # # Tính độ chính xác so với kết quả thực tế
        # accuracy = accuracy_score(self.ytest, y_pred) * 100         
        # # Ma trận nhầm lẫn
        # confusion = confusion_matrix(self.ytest, y_pred)  

        # # Xuất kết quả
        # print(f"Mẫu test: {self.Xtest.shape[0]}")
        # print(f"Độ chính xác: {accuracy}%")  
        # print(f"Confusion Matrix: \n{confusion}")     

        validateNoLib = ValidateNoLib(self.ytest, y_pred)
        print("Số lượng mẫu:", validateNoLib.getSampleSize())
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print("Độ chính xác:", validateNoLib.accuracy())
        for label in range(len(validateNoLib.getSampleClasses())):  
            print(f"Precision cho lớp {label}:", validateNoLib.precision(label))
            print(f"Recall cho lớp {label}:", validateNoLib.recall(label))
            print(f"F-score cho lớp {label}:", validateNoLib.fscore(label))

    def predict(self):
        path='Dataset\predict\\naive-bayes-withlib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()

        # ENCODE DATA Chuyển từ chữ sang số
        label_encoder = LabelEncoder()
        columns = ['Size','Number','Thickness','Lung Cancer']
        for col in columns:
            predictDataframe[col] = label_encoder.fit_transform(predictDataframe[col])

        X = predictDataframe[['Size','Number','Thickness']]
        predictDataframe['Lung Cancer'] = self.model.predict(X)

        # Label decode
        for col in ['Size','Number','Thickness','Lung Cancer']:
            predictDataframe[col] = self.label_encoders[col].inverse_transform(predictDataframe[col])

        csv_handler.write_csv(predictDataframe, path)
        os.startfile(path)
