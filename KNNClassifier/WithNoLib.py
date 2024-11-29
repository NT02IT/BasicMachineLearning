import math
from sklearn.metrics import mean_squared_error
from Normalization.Normalization import Normalization
from Validate.ValidateNoLib import ValidateNoLib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.CSVHandler import CSVHandler

class WithNoLib:
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

        self.trainSet = self.dataframe[:split_index]
        self.testSet = self.dataframe[split_index:]
        self.trainRows, self.trainColumns = self.trainSet.shape
        self.testRows, self.testColumns = self.testSet.shape

        # print("X_train shape:", self.X_train.shape)
        # print("y_train shape:", self.y_train.shape)
        # print("Sample X_train:\n", self.X_train.head())
        # print("Sample y_train:\n", self.y_train.head())

    def euclidean_distance(self, pointA, pointB):
        # tmp = 0
        # for i in range(len(pointA)):  
        #     tmp += (float(pointA.iloc[i]) - float(pointB.iloc[i])) ** 2
        # return math.sqrt(tmp)
        
        pointA = np.array(pointA)
        pointB = np.array(pointB)
        
        # Tính khoảng cách Euclidean
        return np.linalg.norm(pointA - pointB)

    def train(self, min_k=1, max_k=10):
        self.mse_values = []
        self.k_values = range(min_k, max_k + 1)

        for k in self.k_values:
            print(f"\rTraining with k = {k}...", end='', flush=True)
            y_pred = self.predict(self.X_train, self.y_train, self.X_test, k)

            mse = np.mean((self.y_test - y_pred) ** 2)
            self.mse_values.append(mse)

        # Tìm giá trị k tối ưu
        self.optimal_k = self.k_values[np.argmin(self.mse_values)]
        print(f'Optimal k: {self.optimal_k}, Minimum MSE: {min(self.mse_values):.4f}')

        return self.mse_values, self.k_values
    
    def predict(self, X_train, y_train, X_test, k):
        y_pred = []
        for i, test_point in X_test.iterrows():
            print(f"\rPredict for test point {i}...", end='', flush=True)
            distances = []

            # Duyệt qua các điểm trong X_train và tính khoảng cách Euclidean
            for _, train_point in X_train.iterrows():
                # Tính khoảng cách Euclidean
                distance = self.euclidean_distance(test_point, train_point)  # Đảm bảo có 2 đối số
                distances.append(distance)

            # Lấy k chỉ số hàng xóm gần nhất
            neighbors_idx = np.argsort(distances)[:k]
            neighbors_labels = [y_train.iloc[idx] for idx in neighbors_idx]

            # Lấy nhãn phổ biến nhất trong k hàng xóm
            predicted_label = max(set(neighbors_labels), key=neighbors_labels.count)
            y_pred.append(predicted_label)

        return np.array(y_pred)

    def test(self):
        y_pred = self.predict(self.X_train, self.y_train, self.X_test, self.optimal_k)
            
        validateNoLib = ValidateNoLib(self.y_test, y_pred)
        print("Số lượng mẫu test:", validateNoLib.getSampleSize())
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print(f"Độ chính xác: {round(validateNoLib.accuracy()*100,2)}%")