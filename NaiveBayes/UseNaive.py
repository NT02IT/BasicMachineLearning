from collections import defaultdict
import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from Normalization.Normalization import Normalization
from Validate.ValidateNoLib import ValidateNoLib
from utils.CSVHandler import CSVHandler

class UseNaive:
    def __init__(self, datasetURL, train_size=0.5):
        csv_handler = CSVHandler(datasetURL)
        dataframe = csv_handler.read_csv()   

        dataframe, label_encoders = Normalization.encode_dataframe(dataframe)
            
        X = dataframe.iloc[:, :-1]     # Chọn tất cả các hàng và cột trừ cột cuối cùng
        y = dataframe.iloc[:, -1]      # Chọn cột cuối cùng

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * train_size)
        self.X_train = X.iloc[:split_index]  
        self.y_train = y.iloc[:split_index]  
        self.X_test = X.iloc[split_index:]   
        self.y_test = y.iloc[split_index:]   

        # Khởi tạo biến lưu xác suất của các phân lớp và xác suất có điều kiện
        self.class_priors = defaultdict(float)
        self.conditional_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    def calculate_class_prior_probabilities(self, y):
        # Tính xác suất của từng phân lớp trong y (sử dụng sửa lỗi Laplace)
        n_classes = len(y.unique())
        total_samples = len(y)
        
        for class_label in y.unique():
            class_count = sum(y == class_label)
            self.class_priors[class_label] = (class_count + 1) / (total_samples + n_classes)
            
    def calculate_conditional_probabilities(self, X, y):
        # Tính xác suất có điều kiện cho từng đặc trưng ở từng phân lớp
        for feature in X.columns:
            unique_values = X[feature].nunique()  # Số lượng giá trị khác nhau cho mỗi đặc trưng
            for class_label in y.unique():
                # Chọn ra các hàng có nhãn phân lớp class_label
                rows_in_class = X[y == class_label]
                for value in range(unique_values):
                    count_value_in_class = sum(rows_in_class[feature] == value)
                    # Áp dụng sửa lỗi Laplace cho xác suất điều kiện
                    self.conditional_probs[feature][class_label][value] = (count_value_in_class + 1) / (len(rows_in_class) + unique_values)

    def training(self):
        # Các phân lớp: [0 1 2 3]
        # Các giá trị của cột 0 (Size): [0 1 2 3 4]
        # Các giá trị của cột 1 (Number): [0 1 2 3 4]
        # Các giá trị của cột 2 (Thinkness): [0 1 2]

        # Đối với cột 0: 
            # Tính lần lượt các xác xuất để 1 item (nhớ sử dụng phép sửa lỗi Laplace)
            # có giá trị là 0 ở phân lớp 0
            # có giá trị là ... ở phân lớp 0
            # có giá trị là 4 ở phân lớp 0
            # tương tự với các phân lớp 1 2 3
        # Tương tự với các cột 1 2

        # Tính các tỉ lệ mẫu thuộc phân lớp 0, 1, 2, 3 (nhớ sử dụng phép sửa lỗi Laplace)

        self.calculate_class_prior_probabilities(self.y_train)
        self.calculate_conditional_probabilities(self.X_train, self.y_train)

    def testing(self):
        y_pred = self.predict(self.X_test)
        validateNoLib = ValidateNoLib(self.y_test, y_pred)
        print("Số lượng mẫu test:", validateNoLib.getSampleSize())
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print(f"Độ chính xác: {round(validateNoLib.accuracy()*100,2)}%")
        # for label in range(len(validateNoLib.getSampleClasses())):  
        #     print(f"Precision cho lớp {label}:", validateNoLib.precision(label))
        #     print(f"Recall cho lớp {label}:", validateNoLib.recall(label))
        #     print(f"F-score cho lớp {label}:", validateNoLib.fscore(label))

    def predict(self, input_data):
        # for col, encoder in self.label_encoders.items():
        #     print(f"Classes trong encoder của cột '{col}': {encoder.classes_}")

        # Dự đoán phân lớp cho từng mẫu
        predictions = []
        for _, row in input_data.iterrows():
            best_class, best_prob = None, -1
            for class_label in self.class_priors:
                # Tính toán xác suất cho phân lớp hiện tại
                prob = self.class_priors[class_label]
                for feature, value in zip(input_data.columns, row):
                    prob *= self.conditional_probs[feature][class_label].get(value, 1e-9)  # 1e-9 để tránh xác suất 0
                # Lưu lại phân lớp có xác suất cao nhất
                if prob > best_prob:
                    best_class, best_prob = class_label, prob
            predictions.append(best_class)

        y_pred = pd.DataFrame(predictions)
        return y_pred