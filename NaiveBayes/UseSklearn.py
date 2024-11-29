import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from Normalization.Normalization import Normalization
from utils.CSVHandler import CSVHandler
from Validate.ValidateNoLib import ValidateNoLib

class UseSklearn:
    def __init__(self, datasetURL, train_size=0.5):
        self.model = GaussianNB()

        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()   

        dataframe, self.label_encoders = Normalization.encode_dataframe(dataframe)
            
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

    def testing(self):
        # Dự đoán trên tập testing
        y_pred = self.model.predict(self.X_test)

        validateNoLib = ValidateNoLib(self.y_test, y_pred)
        print("Số lượng mẫu test:", validateNoLib.getSampleSize())
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print(f"Độ chính xác: {round(validateNoLib.accuracy()*100, 2)}%")

    def predict(self, input_data):
        return self.model.predict(input_data)
