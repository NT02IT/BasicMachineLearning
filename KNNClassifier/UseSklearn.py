from sklearn.metrics import mean_squared_error
from Normalization.Normalization import Normalization
from Validate.ValidateNoLib import ValidateNoLib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from utils.CSVHandler import CSVHandler

class UseSklearn:
    def __init__(self, datasetURL, train_size=0.5):
        csv_handler = CSVHandler(datasetURL)
        self.dataframe = csv_handler.read_csv()   

        self.dataframe, self.label_encoders = Normalization.encode_dataframe(self.dataframe)
            
        X = self.dataframe.iloc[:, :-1]     # Chọn tất cả các hàng và cột trừ cột cuối cùng
        y = self.dataframe.iloc[:, -1]      # Chọn cột cuối cùng

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(self.dataframe) * train_size)
        self.X_train = X.iloc[:split_index]  
        self.y_train = y.iloc[:split_index]  
        self.X_test = X.iloc[split_index:]   
        self.y_test = y.iloc[split_index:]  

        # print("X_train shape:", self.X_train.shape)
        # print("y_train shape:", self.y_train.shape)
        # print("Sample X_train:\n", self.X_train.head())
        # print("Sample y_train:\n", self.y_train.head())

    def train(self, min_k=1, max_k=30):
        self.mse_values = []
        self.k_values = range(min_k, max_k + 1)

        for k in self.k_values:
            print(f"\rTraining with k = {k}...", end='', flush=True)
            # Tạo model KNN với k láng giềng
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.model.fit(self.X_train, self.y_train)

            # Dự đoán trên tập test
            y_pred = self.model.predict(self.X_test)

            # Tính MSE và lưu lại
            mse = mean_squared_error(self.y_test, y_pred)
            self.mse_values.append(mse)

        # Tìm giá trị k tối ưu
        optimal_k = self.k_values[self.mse_values.index(min(self.mse_values))]
        print(f'\r\nOptimal k: {optimal_k}, Minimum MSE: {min(self.mse_values):.4f}', end='', flush=True)
        self.model = KNeighborsClassifier(optimal_k)
        self.model.fit(self.X_train, self.y_train)

        return self.mse_values, self.k_values
    
    def predict(self, input_data):
        return self.model.predict(input_data)

    def test(self):
        y_pred = self.predict(self.X_test)

        # Lấy LabelEncoder cho cột cuối cùng
        # last_column_name = self.dataframe.columns[-1]
        # last_column_encoder = self.label_encoders[last_column_name]

        # y_true_decode = last_column_encoder.inverse_transform(self.y_test)
        # y_pred_decode = last_column_encoder.inverse_transform(y_pred)

        # for true, pred in zip(y_true_decode, y_pred_decode):
        #     print(f"label: {true} -> predicted: {pred}")
            
        validateNoLib = ValidateNoLib(self.y_test, y_pred)
        print("Số lượng mẫu test:", validateNoLib.getSampleSize())
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print(f"Độ chính xác: {round(validateNoLib.accuracy()*100,2)}%")
        # print(f"Mean Squared Error: {round(mean_squared_error(self.y_test, y_pred), 4)}")
