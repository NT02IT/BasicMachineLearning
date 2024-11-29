import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from utils.CSVHandler import CSVHandler
import os

class UseSklearn:
    def __init__(self, datasetURL, train_size=0.5):
        self.model = SGDRegressor(max_iter=1500, warm_start=True, learning_rate="constant", eta0=1e-8, random_state=27)
        # Lists để lưu loss
        self.iterations = []
        self.loss_values = []

        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()   

        # Chuẩn hóa dữ liệu từ string sang số "1,2" -> 1.2
        for col in dataframe.columns:
            if dataframe[col].dtype not in ['int64', 'float64']:
                dataframe[col] = dataframe[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
            
        X = dataframe.iloc[:, :-1]     # Chọn tất cả các hàng và cột trừ cột cuối cùng
        y = dataframe.iloc[:, -1]      # Chọn cột cuối cùng

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * train_size)
        self.X_train = X.iloc[:split_index]  
        self.y_train = y.iloc[:split_index]  
        self.X_test = X.iloc[split_index:]   
        self.y_test = y.iloc[split_index:]   

    def train(self):
        # self.model.fit(self.X_train, self.y_train)  
        # epochs = 1500  # Số vòng lặp
        count_patience = 0
        for i in range(self.model.max_iter):
            self.model.partial_fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_train)
            loss = mean_squared_error(self.y_train, y_pred)
            self.loss_values.append(loss)
            self.iterations.append(i + 1)

            # Kiểm tra dừng sớm
            threshold = 1e+5  # Ngưỡng thay đổi loss
            if i > 0 and abs(self.loss_values[-1] - self.loss_values[-2]) < threshold:
                count_patience += 1
                if count_patience >= 50: 
                    print(f"Dừng tại vòng lặp {i} vì loss không thay đổi đáng kể")
                    break
            else:
                count_patience = 0
        return self.loss_values

    def predict(self, data_input):    
        self.y_pred = self.model.predict(data_input)
        return self.y_pred
    
    def predictFor(self, data_input):
        data_input = np.array(data_input)
        # Chuyển data_input thành vector hàng (1, n)
        if data_input.ndim == 1:
            data_input = data_input.reshape(1, -1)
        y_pred = self.model.predict(data_input)
        return y_pred
    
    def test(self):
        predictions = self.predict(self.X_test)
        loss = mean_squared_error(self.y_test, predictions)
        return loss

    def getModelInfo(self):
        print("Mẫu training:", len(self.y_train))
        print("Hệ số tự do (Intercept):", self.model.intercept_)
        print("Hệ số của các biến độc lập (Coefficients):", self.model.coef_)

    def plot_loss(self):
        # Vẽ biểu đồ hàm loss qua các vòng lặp
        plt.plot(range(len(self.loss_values)), self.loss_values)
        plt.xlabel('Số lần lặp (Iterations)')
        plt.ylabel('Loss (MSE)')
        plt.title('Sự thay đổi của MSE theo các vòng lặp Gradient Descent')
        plt.show()

    
    