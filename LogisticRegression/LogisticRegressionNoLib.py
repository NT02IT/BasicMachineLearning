import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.CSVHandler import CSVHandler

class LogisticRegressionNoLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\logistic-regression.csv')
        dataframe = csv_handler.read_csv()
        self.X = dataframe[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']]
        self.y = dataframe['5-year survival']

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * 0.5)
        self.Xtrain = self.X.iloc[:split_index]  
        self.ytrain = self.y.iloc[:split_index]  
        self.Xtest = self.X.iloc[split_index:]   
        self.ytest = self.y.iloc[split_index:]       

    def training(self):
        self._gradient_descent(self.Xtrain, self.ytrain)

    def trainingFullDataset(self):
        self._gradient_descent(self.X, self.y)

    def testing(self):
        # Dự đoán trên tập testing
        linear_model = np.dot(self.Xtest, self.weights) + self.bias
        y_hat = self._sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_hat]  # Ngưỡng 0.5

        # Tính độ chính xác so với kết quả thực tế
        matches = self.ytest == predictions        
        accuracy = matches.mean() * 100  # tính trung bình và chuyển sang %
        print(f"Độ chính xác: {accuracy}%")
        

    def predict(self):
        path='Dataset\predict\logistic-regression-nolib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()
        X = predictDataframe[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']]
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]  # Ngưỡng 0.5
        predictDataframe['5-year survival'] = predictions
        csv_handler.write_csv(predictDataframe, path)
        os.startfile(path)
    
    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.bias)
        print("Coefficients (hệ số của các biến độc lập):", self.weights)

    def _sigmoid(self, z):
        z = np.clip(z, -709, 709)  # Giới hạn ở -709 và 709 để tránh khi e mũ quá lớn hoặc quá nhỏ sẽ gây lỗi tràn số
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y, y_hat):
        # Tính toán chi phí (loss) sử dụng hàm binary cross-entropy.
        m = self.ytrain.shape[0]  # Số lượng mẫu
        cost = -1/m * np.sum(y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10))
        return cost
    
    def _gradient_descent(self, X, y, learning_rate=0.01, iterations=1500):
        m, n = X.shape  # m: số lượng mẫu, n: số lượng đặc trưng
        # Khởi tạo trọng số
        self.weights = np.zeros(n)  
        self.bias = 0     
        
        for i in range(iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_model)

            # Tính toán chi phí giữa y dự đoán và y thực tế
            cost = self._compute_cost(y, y_hat)

            # Tính toán gradient
            dw = (1/m) * np.dot(X.T, (y_hat - y))  # Gradient cho trọng số
            db = (1/m) * np.sum(y_hat - y)  # Gradient cho bias

            # Cập nhật trọng số và bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if cost < 1e-6:
                print(f"Dừng tại vòng lặp {i} với chi phí = {cost:.6f}")
                break