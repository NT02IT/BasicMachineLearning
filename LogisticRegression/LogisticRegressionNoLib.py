import numpy as np
import pandas as pd

class LogisticRegressionNoLib:
    def __init__(self, dataframe):
        self.dataframe = dataframe  

    def training(self):
        self.X = self.dataframe[['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5']]
        self.y = self.dataframe['5-year survival']
        self.gradient_descent()

    def predict(self, X):
        input_array = pd.DataFrame([X], columns=['Gene1', 'Gene2', 'Gene3', 'Gene4', 'Gene5'])
        linear_model = np.dot(input_array, self.weights) + self.bias
        y_hat = self.sigmoid(linear_model)
        predictions = [1 if i > 0.5 else 0 for i in y_hat]  # Ngưỡng 0.5
        return np.array(predictions)
    
    def getModelInfo(self):
        print("Intercept (hệ số tự do):", self.bias)
        print("Coefficients (hệ số của các biến độc lập):", self.weights)

    def sigmoid(self, z):
        z = np.clip(z, -709, 709)  # Giới hạn ở -709 và 709 để tránh khi e mũ quá lớn hoặc quá nhỏ sẽ gây lỗi tràn số
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y, y_hat):
        # Tính toán chi phí (loss) sử dụng hàm binary cross-entropy.
        m = self.y.shape[0]  # Số lượng mẫu
        cost = -1/m * np.sum(y * np.log(y_hat + 1e-10) + (1 - y) * np.log(1 - y_hat + 1e-10))
        return cost
    
    def gradient_descent(self, learning_rate=0.01, iterations=1000):
        # Thuật toán Gradient Descent để điều chỉnh trọng số cho Logistic Regression.
        m, n = self.X.shape  # m: số lượng mẫu, n: số lượng đặc trưng
        self.weights = np.zeros(n)  # Khởi tạo trọng số
        self.bias = 0 

        for i in range(iterations):
            linear_model = np.dot(self.X, self.weights) + self.bias
            y_hat = self.sigmoid(linear_model)

            # Tính toán chi phí giữa y dự đoán và y thực tế
            cost = self.compute_cost(self.y, y_hat)

            # Tính toán gradient
            dw = (1/m) * np.dot(self.X.T, (y_hat - self.y))  # Gradient cho trọng số
            db = (1/m) * np.sum(y_hat - self.y)  # Gradient cho bias

            # Cập nhật trọng số và bias
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db