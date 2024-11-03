from collections import defaultdict
import os
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from Validate.ValidateNoLib import ValidateNoLib
from utils.CSVHandler import CSVHandler

class NaiveBayesNoLib:
    def __init__(self):
        self.model = GaussianNB()

        csv_handler = CSVHandler('Dataset\\naive-bayes.csv')
        dataframe = csv_handler.read_csv()

        # ENCODE DATA Chuyển từ chữ sang số
        columns = ['Size','Number','Thickness','Lung Cancer']
        self.label_encoders = {}  # Tạo từ điển lưu LabelEncoder cho mỗi cột
        for col in columns:
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
            self.label_encoders[col] = le  # Lưu LabelEncoder để chuyển đổi ngược lại

        self.X = dataframe[['Size','Number','Thickness']]
        self.y = dataframe['Lung Cancer']

        # Chia dữ liệu làm 2 phần training và testing
        split_index = int(len(dataframe) * 0.8)
        self.Xtrain = self.X.iloc[:split_index]  
        self.ytrain = self.y.iloc[:split_index]  
        self.Xtest = self.X.iloc[split_index:]   
        self.ytest = self.y.iloc[split_index:]      

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

        # Tính các xác suất của phân lớp và xác suất có điều kiện
        self.calculate_class_prior_probabilities(self.ytrain)
        self.calculate_conditional_probabilities(self.Xtrain, self.ytrain)

    def trainingFullDataset(self):
        self.calculate_class_prior_probabilities(self.y)
        self.calculate_conditional_probabilities(self.X, self.y)

    def testing(self):
        # Dự đoán phân lớp cho từng mẫu
        predictions = []
        for _, row in self.Xtest.iterrows():
            best_class, best_prob = None, -1
            for class_label in self.class_priors:
                # Tính toán xác suất cho phân lớp hiện tại
                prob = self.class_priors[class_label]
                for feature, value in zip(self.Xtest.columns, row):
                    prob *= self.conditional_probs[feature][class_label].get(value, 1e-9)  # 1e-9 để tránh xác suất 0
                # Lưu lại phân lớp có xác suất cao nhất
                if prob > best_prob:
                    best_class, best_prob = class_label, prob
            predictions.append(best_class)

        predictions_df = pd.DataFrame(predictions, columns=['Lung Cancer'])
        validateNoLib = ValidateNoLib(self.ytest, predictions_df)
        print("Số lượng mẫu test:", validateNoLib.getSampleSize())
        print("Các phân lớp:", validateNoLib.getSampleClasses())
        print("Ma trận nhầm lẫn:\n", validateNoLib.confusionMatrix())
        print("Độ chính xác:", validateNoLib.accuracy())
        # for label in range(len(validateNoLib.getSampleClasses())):  
        #     print(f"Precision cho lớp {label}:", validateNoLib.precision(label))
        #     print(f"Recall cho lớp {label}:", validateNoLib.recall(label))
        #     print(f"F-score cho lớp {label}:", validateNoLib.fscore(label))

    def predict(self):
        path='Dataset\predict\\naive-bayes-nolib.csv'       
        csv_handler = CSVHandler(path)
        predictDataframe = csv_handler.read_csv()

        # for col, encoder in self.label_encoders.items():
        #     print(f"Classes trong encoder của cột '{col}': {encoder.classes_}")

        # Lặp qua từng cột và dùng label_encoders để mã hóa
        for col in self.label_encoders:
            if col in predictDataframe.columns:
                predictDataframe[col] = self.label_encoders[col].transform(predictDataframe[col])

        X = predictDataframe[['Size','Number','Thickness']]

        # Dự đoán phân lớp cho từng mẫu
        predictions = []
        for _, row in X.iterrows():
            best_class, best_prob = None, -1
            for class_label in self.class_priors:
                # Tính toán xác suất cho phân lớp hiện tại
                prob = self.class_priors[class_label]
                for feature, value in zip(X.columns, row):
                    prob *= self.conditional_probs[feature][class_label].get(value, 1e-9)  # 1e-9 để tránh xác suất 0
                # Lưu lại phân lớp có xác suất cao nhất
                if prob > best_prob:
                    best_class, best_prob = class_label, prob
            predictions.append(best_class)

        for col in ['Size','Number','Thickness','Lung Cancer']:
            predictDataframe[col] = self.label_encoders[col].inverse_transform(predictDataframe[col])

        csv_handler.write_csv(predictDataframe, path)
        os.startfile(path)