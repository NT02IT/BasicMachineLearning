import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from Normalization.Normalization import Normalization
from utils.CSVHandler import CSVHandler

class UseGradientDescent:
    def __init__(self, datasetURL, train_size=0.5):
        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()   

        # Chuẩn hóa dữ liệu từ string sang số "1,2" -> 1.2
        pattern = r"^\d,\d$" 
        for col in dataframe.columns:
            if dataframe[col].dtype not in ['int64', 'float64']:
                if dataframe[col].str.match(pattern).all():
                    dataframe[col] = dataframe[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

        dataframe = Normalization.minMaxNormalizationNolib(dataframe, 0, 1)
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
        cost_values = self._gradient_descent(self.X_train, self.y_train)
        cost_values_validate = self._gradient_descent(self.X_test, self.y_test)
        return cost_values, cost_values_validate

    def predict(self, data_input):
        linear_model = np.dot(data_input, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_pred]  # Ngưỡng phân lớp 0.5
        return predictions
    
    def predictFor(self, data_input):
        data_input = np.array(data_input)
        # Chuyển data_input thành vector hàng (1, n)
        if data_input.ndim == 1:
            data_input = data_input.reshape(1, -1)
        data_input = pd.DataFrame(data_input, columns=self.X_train.columns.tolist())
        data_input = Normalization.encode_dataframe_with_encoders(data_input, self.label_encoders)
        data_input_np = self.convert_to_numeric(data_input)

        linear_model = np.dot(data_input_np, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model[0])
        prediction = 1 if y_pred > 0.5 else 0 # Ngưỡng phân lớp 0.5
        
        return prediction

    def testing(self):
        # Dự đoán trên tập testing
        linear_model = np.dot(self.X_test, self.weights) + self.bias
        y_hat = self._sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_hat]  # Ngưỡng 0.5

        # Tính độ chính xác so với kết quả thực tế
        matches = self.y_test == predictions        
        accuracy = matches.mean() * 100  # tính trung bình và chuyển sang %
        print(f"Độ chính xác: {round(accuracy, 2)}%")    
    
    def getModelInfo(self):
        print("Mẫu training:", len(self.y_train))
        print("Intercept (hệ số tự do):", self.bias)
        print("Coefficients (hệ số của các biến độc lập):", self.weights)
        print("MSE:", self.cost_values[-1])
        # self.plot_loss(self.cost_values) 
        

    def _gradient_descent(self, X, y, learning_rate=1e-4, iterations=1500):
        m, n = X.shape  # m: số lượng mẫu, n: số lượng đặc trưng
        # Khởi tạo trọng số
        self.weights = np.zeros(n)  
        self.bias = 0     
        
        count_patience = 0
        self.cost_values = []
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

            self.cost_values.append(cost)
            threshold = 10  # Ngưỡng thay đổi loss
            if i > 0 and abs(self.cost_values[-1] - self.cost_values[-2]) < threshold:
                count_patience += 1
                if count_patience >= 50: 
                    print(f"Dừng tại vòng lặp {i} vì loss không thay đổi đáng kể")
                    break
            else:
                count_patience = 0   
        return self.cost_values
    
    def _sigmoid(self, z):
        # Giới hạn ở -709 và 709 để tránh khi e mũ quá lớn hoặc quá nhỏ sẽ gây lỗi tràn số
        z = np.clip(z, -709, 709)  
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y, y_hat):
        # Tính toán chi phí (cost) sử dụng hàm binary cross-entropy.
        m = self.y_train.shape[0]  # Số lượng mẫu
        cost = -1/m * np.sum(y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10))
        return cost
        
    def plot_loss(self, loss_values):
        # Vẽ biểu đồ thể hiện sự thay đổi của MSE theo thời gian (iterations)
        plt.plot(range(len(loss_values)), loss_values)
        plt.xlabel('Số lần lặp (Iterations)')
        plt.ylabel('Cost (binary cross-entropy)')
        plt.title('Sự thay đổi của binary cross-entropy theo các vòng lặp Gradient Descent')
        plt.show()

    # Hàm chuyển đổi dữ liệu từ chuỗi sang kiểu số 
    def convert_to_numeric(self, data_input):
        data_input = np.array(data_input)
        for i in range(data_input.shape[1]):  
            try:
                data_input[:, i] = data_input[:, i].astype(float)
            except ValueError:
                pass
        return data_input