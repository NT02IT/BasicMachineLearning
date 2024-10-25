import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.CSVHandler import CSVHandler
import os

# Model: y(x1, x2, x3) = w0 + w1.x1 + w2.x2 + w3.x3
# X = [1, x1(1:N), x2(1:N), x3(1:N)]^T
# w = [w0, w1, w2, w3]^T
# y = X.w
# Loss function: J = sum((w0 + w1.x1[i] + w2.x2[i] + w3.x3[i] - y[i])^2)/N
# Đạo hàm Loss function theo các biến w0, w1, w2, w3
#   - w0: N.w0 + w1.sum(x1[i]) + w2.sum(x2[i]) + w3.sum(x3[i]) - sum(y[i])
#   - w1: w0.sum(x1[i]) + w1.sum(x1[i]^2) + w2.sum(x1[i].x2[i]) + w3.sum(x1[i].x3[i]) - sum(y[i])
#   - w2: w0.sum(x2[i]) + w1.sum(x2[i]^2) + w2.sum(x2[i].x1[i]) + w3.sum(x2[i].x3[i]) - sum(y[i])
#   - w3: w0.sum(x3[i]) + w1.sum(x3[i]^2) + w2.sum(x3[i].x1[i]) + w3.sum(x3[i].x2[i]) - sum(y[i])
# Giải hệ phương trình trên để tìm được w0, w1, w2, w3

class LinearRegressionNoLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\linear-regression.csv')
        self.dataframe = csv_handler.read_csv()
    
    def training(self):
        #LẤY DỮ LIỆU TRÊN BẢNG
        N = self.dataframe.values.shape[0]   # (0) Số hàng trong bảng
        x1 = self.dataframe['Temperature'].values.reshape(-1, 1)
        x2 = self.dataframe['Tourists'].values.reshape(-1, 1)
        x3 = self.dataframe['SunnyDays'].values.reshape(-1, 1)
        y = self.dataframe['PredictedSales'].values.reshape(-1, 1)

        # Xây dựng ma trận X
        X = np.hstack((np.ones((N, 1)), x1, x2, x3))

        #XÂY DỰNG MA TRẬN HỆ SỐ ĐỂ GIẢI HỆ PHƯƠNG TRÌNH (CÓ ĐƯỢC SAU KHI ĐẠO HÀM TỪNG PHẦN)
        #Ma trận hệ số 
        A = np.array([[N, x1.sum(), x2.sum(), x3.sum()],
            [x1.sum(), np.sum(x1**2), np.sum(x1*x2), np.sum(x1*x3)],
            [x2.sum(), np.sum(x2*x1), np.sum(x2**2), np.sum(x2*x3)],
            [x3.sum(), np.sum(x3*x1), np.sum(x3*x2), np.sum(x3**2)]])
        B = np.array([y.sum(), np.sum(y*x1), np.sum(y*x2), np.sum(y*x3)])
        
        W = np.linalg.solve(A, B)
        self.W = np.array([W[0], W[1], W[2], W[3]]).reshape(-1, 1)        

    def predict(self):
        path='Dataset\predict\linear-regression-nolib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()
        predictDataframe['PredictedSales'] = self.W[0] + self.W[1]*predictDataframe['Temperature'] + self.W[2]*predictDataframe['Tourists'] + self.W[3]*predictDataframe['SunnyDays']
        csv_handler.write_csv(predictDataframe, path)
        os.startfile(path)
    
    def getModelInfo(self):
        print(f"Intercept (hệ số tự do): {self.W[0]}")
        print(f"Coefficients (hệ số của các biến độc lập): [{self.W[1]} {self.W[2]} {self.W[3]}]")