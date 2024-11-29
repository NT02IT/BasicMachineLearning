import math
import threading
from sklearn.metrics import mean_squared_error
from Normalization.Normalization import Normalization
from Validate.ValidateNoLib import ValidateNoLib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from collections import Counter

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

        self.mse_values = []
        self.k_values = []

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
    
    def train_with_k(self, k, min_k):
        print(f"\rTraining with k = {k}...", end='', flush=True)
        y_pred = self.predict(self.X_train, self.y_train, self.X_test, k)
        mse = np.mean((self.y_test - y_pred) ** 2)

        self.mse_values[k - min_k] = mse

    def train(self, min_k=1, max_k=10):
        self.mse_values = [None] * (max_k - min_k + 1)  # Khởi tạo danh sách chứa MSE cho mỗi k
        self.k_values = range(min_k, max_k + 1)

        threads = []
        
        # Tạo các luồng và bắt đầu chúng
        for k in self.k_values:
            thread = threading.Thread(target=self.train_with_k, args=(k, min_k))
            threads.append(thread)
            thread.start()
            # self.train_with_k(k, min_k)
        
        # Đợi tất cả các luồng kết thúc
        for thread in threads:
            thread.join()

        # Tìm giá trị k tối ưu
        if None not in self.mse_values:
            self.optimal_k = self.k_values[np.argmin(self.mse_values)]
            print(f'\r\nOptimal k: {self.optimal_k}, Minimum MSE: {min(self.mse_values):.4f}', end='', flush=True)

        return self.mse_values, self.k_values
    
    # def predict(self, X_train, y_train, X_test, k): # Qúa chậm
    #     y_pred = []
    #     for i, test_point in X_test.iterrows():
    #         print(f"\rPredict for test point {i}...", end='', flush=True)
    #         distances = []

    #         # Duyệt qua các điểm trong X_train và tính khoảng cách Euclidean
    #         for _, train_point in X_train.iterrows():
    #             # Tính khoảng cách Euclidean
    #             distance = self.euclidean_distance(test_point, train_point)  # Đảm bảo có 2 đối số
    #             distances.append(distance)

    #         # Lấy k chỉ số hàng xóm gần nhất
    #         neighbors_idx = np.argsort(distances)[:k]
    #         neighbors_labels = [y_train.iloc[idx] for idx in neighbors_idx]

    #         # Lấy nhãn phổ biến nhất trong k hàng xóm
    #         predicted_label = max(set(neighbors_labels), key=neighbors_labels.count)
    #         y_pred.append(predicted_label)

    #     return np.array(y_pred)

    def predict(self, X_train, y_train, X_test, k):
        X_train_np = X_train.to_numpy()
        X_test_np = X_test.to_numpy()
        y_train_np = y_train.to_numpy()

        # Tính toán ma trận khoảng cách Euclidean giữa X_test và X_train
        distances = cdist(X_test_np, X_train_np, metric='euclidean')

        y_pred = []
        for i, dist in enumerate(distances):
            print(f"\rPredict for test point {i}...", end='', flush=True)

            # Lấy k chỉ số hàng xóm gần nhất
            neighbors_idx = np.argsort(dist)[:k]
            neighbors_labels = y_train_np[neighbors_idx]

            # Lấy nhãn phổ biến nhất trong k hàng xóm
            predicted_label = Counter(neighbors_labels).most_common(1)[0][0]
            y_pred.append(predicted_label)
        return np.array(y_pred)

    def test(self):
        y_pred = self.predict(self.X_train, self.y_train, self.X_test, self.optimal_k)
            
        validateNoLib = ValidateNoLib(self.y_test, y_pred)
        print("\rSố lượng mẫu test:", validateNoLib.getSampleSize(), end='', flush=True)
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print(f"Độ chính xác: {round(validateNoLib.accuracy()*100,2)}%")