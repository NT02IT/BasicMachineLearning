import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss

from Normalization.Normalization import Normalization
from utils.CSVHandler import CSVHandler

class UseSklearn:
    def __init__(self, datasetURL, train_size=0.5):
        self.model = SGDClassifier(loss='log_loss', shuffle=False, random_state=42, max_iter=1500, tol=1e-2, alpha=0.1, penalty='l2') #

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
        # self.model.fit(self.X_train, self.y_train)
        self.loss_history = []
        self.loss_history_validate = []
        count_patience = 0 
        for i in range(self.model.max_iter):
            self.model.partial_fit(self.X_train, self.y_train, classes=np.unique(self.y_train))
            # Dự đoán trên tập huấn luyện để tính toán loss
            y_train_pred = self.model.predict_proba(self.X_train)  
            loss = log_loss(self.y_train, y_train_pred)  # Tính loss
            self.loss_history.append(loss)

            y_validate_pred = self.model.predict_proba(self.X_test) 
            loss_validate = log_loss(self.y_test, y_validate_pred)  
            self.loss_history_validate.append(loss_validate)

            threshold = 10  # Ngưỡng thay đổi loss
            if i > 0 and abs(self.loss_history[-1] - self.loss_history[-2]) < threshold:
                count_patience += 1
                if count_patience >= 50: 
                    print(f"Dừng tại vòng lặp {i} vì loss không thay đổi đáng kể")
                    break
            else:
                count_patience = 0 
        return self.loss_history, self.loss_history_validate

    def predict(self, data_input):
        return self.model.predict(data_input)
    
    def predictFor(self, data_input):
        data_input = np.array(data_input)
        # Chuyển data_input thành vector hàng (1, n)
        if data_input.ndim == 1:
            data_input = data_input.reshape(1, -1)
        data_input = pd.DataFrame(data_input, columns=self.X_train.columns.tolist())
        data_input = Normalization.encode_dataframe_with_encoders(data_input, self.label_encoders)
        y_pred = self.model.predict(data_input)
        
        return y_pred
    

    def testing(self):
        # Dự đoán trên tập testing
        y_pred = self.model.predict(self.X_test)

        # Tính độ chính xác so với kết quả thực tế
        accuracy = accuracy_score(self.y_test, y_pred) * 100
        print(f"Độ chính xác: {round(accuracy, 2)}%")    
    
    def getModelInfo(self):
        print("Mẫu training:", len(self.y_train))
        print("Intercept (hệ số tự do):", self.model.intercept_[0])
        print("Coefficients (hệ số của các biến độc lập):", self.model.coef_[0])