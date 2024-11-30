import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Normalization.Normalization import Normalization
from utils.CSVHandler import CSVHandler

class UseGradientDescent:
    def __init__(self, datasetURL, train_size=0.5, learning_rate=0.01, iterations=1500): # Thử thêm với learning_rate=1e-4
        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()
        dataframe = Normalization.minMaxNormalizationWlib(dataframe, 0, 1) 
        for col in dataframe.columns:
            if dataframe[col].dtype not in ['int64', 'float64']:
                dataframe[col] = dataframe[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)

        X = dataframe.iloc[:, :-1].values  # Chọn tất cả các hàng và cột trừ cột cuối cùng
        y = dataframe.iloc[:, -1].values  # Chọn cột cuối cùng

        # Thêm cột 1 vào X để tính toán hệ số tự do (bias)
        X = np.c_[np.ones(X.shape[0]), X]  # Thêm cột 1 cho hệ số tự do (bias)

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * train_size)
        self.X_train = X[:split_index]  
        self.y_train = y[:split_index]  
        self.X_test = X[split_index:]   
        self.y_test = y[split_index:]   

        # Khởi tạo các tham số cho Gradient Descent
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = np.zeros(self.X_train.shape[1])  # Khởi tạo trọng số (w) với giá trị 0

    def train(self):
        # Gradient Descent để tối ưu trọng số
        m = len(self.X_train)
        loss_values = []
        loss_values_validate = []
        count_patience = 0
        for i in range(self.iterations):
            predictions = self.X_train.dot(self.weights)  # Dự đoán y = Xw
            errors = predictions - self.y_train  # Tính sai số (error)

            predictions_validate = self.X_test.dot(self.weights)

            # Cập nhật trọng số bằng cách sử dụng Gradient Descent
            gradient = (2/m) * self.X_train.T.dot(errors)  # Gradient of MSE
            gradient = np.clip(gradient, -1e10, 1e10)  # Giới hạn giá trị gradient trong phạm vi hợp lý tránh lỗi tràn số hoặc NaN
            self.weights -= self.learning_rate * gradient  # Cập nhật trọng số

            # Tính MSE cho mỗi lần lặp
            loss = self._meanSquaredError(self.y_train, predictions)
            loss_values.append(loss)

            loss_validate = self._meanSquaredError(self.y_test, predictions_validate)
            loss_values_validate.append(loss_validate)

            threshold = 1e+5  # Ngưỡng thay đổi loss
            if i > 0 and abs(loss_values[-1] - loss_values[-2]) < threshold:
                count_patience += 1
                if count_patience >= 50: 
                    print(f"Dừng tại vòng lặp {i} vì loss không thay đổi đáng kể")
                    break
            else:
                count_patience = 0
        return loss_values, loss_values_validate

    def predict(self, data_input):
        # Dự đoán giá trị đầu ra cho dữ liệu đầu vào
        return data_input.dot(self.weights)
    
    def predictFor(self, data_input):
        data_input = np.array(data_input)
        # Chuyển data_input thành vector hàng (1, n)
        if data_input.ndim == 1:
            data_input = data_input.reshape(1, -1)
        # Thêm cột hệ số tự do (bias term) vào đầu mảng
        data_input = np.c_[np.ones(data_input.shape[0]), data_input]
        # Dự đoán y = X * theta
        y_pred = data_input.dot(self.weights)
        return y_pred

    def test(self):
        # Dự đoán trên tập kiểm tra và tính MSE
        predictions = self.predict(self.X_test)
        loss = self._meanSquaredError(self.y_test, predictions)
        return loss

    def getModelInfo(self):
        # In ra thông tin về mô hình
        print("Số lượng mẫu training:", len(self.y_train))
        print("Hệ số tự do (Intercept):", self.weights[0])
        print("Hệ số của các biến độc lập (Coefficients):", self.weights[1:])

    def plot_loss(self, loss_values):
        # Vẽ biểu đồ thể hiện sự thay đổi của MSE theo thời gian (iterations)
        plt.plot(range(len(loss_values)), loss_values)
        plt.xlabel('Số lần lặp (Iterations)')
        plt.ylabel('Loss (MSE)')
        plt.title('Sự thay đổi của MSE theo các vòng lặp Gradient Descent')
        plt.show()

    def _meanSquaredError(self, y_true, y_pred):
        errors = y_true - y_pred   
        errors = np.clip(errors, -1e10, 1e10)  # Giới hạn giá trị lỗi trong phạm vi hợp lý tránh lỗi tràn số  
        squared_errors = errors ** 2    # Tính bình phương của các lỗi     
        mse = np.mean(squared_errors)   # Tính trung bình các bình phương lỗi        
        return mse
