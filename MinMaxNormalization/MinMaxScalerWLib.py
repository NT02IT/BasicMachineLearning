import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.CSVHandler import CSVHandler

class MinMaxScalerWLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\\min-max-normalization.csv')
        self.dataframe = csv_handler.read_csv()

    def Normalization(self):
        # Chuyển đổi các dấu '?' thành NaN để dễ xử lý
        self.dataframe.replace('?', pd.NA, inplace=True)

        # Chuyển đổi các cột có số liệu thành kiểu số và xử lý missing data
        num_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
        self.dataframe[num_cols] = self.dataframe[num_cols].apply(pd.to_numeric, errors='coerce')

        # Điền giá trị trung bình của mỗi cột cho các ô thiếu dữ liệu
        self.dataframe[num_cols] = self.dataframe[num_cols].fillna(self.dataframe[num_cols].mean())

        scaler = MinMaxScaler(feature_range=(0, 100))
        self.dataframe[num_cols] = scaler.fit_transform(self.dataframe[num_cols])

        # Thực hiện xóa các giá trị trùng lặp trong mỗi cột
        for col in self.dataframe.columns:
            self.dataframe[col] = self.dataframe[col].where(~self.dataframe[col].duplicated(), np.nan)

        # Dịch chuyển các dữ liệu khác NAN lên đầu cột
        for col in self.dataframe.columns:
            non_na = self.dataframe[col].dropna()
            na_values = self.dataframe[col][self.dataframe[col].isna()]
            self.dataframe[col] = pd.concat([non_na, na_values], axis=0, ignore_index=True)

        path='Dataset\predict\min-max-normalization-withlib.csv'
        csvHandler = CSVHandler(path)
        csvHandler.write_csv(self.dataframe, path)
        os.startfile(path)