import os
import numpy as np
import pandas as pd
from utils.CSVHandler import CSVHandler

class MinMaxScalerNoLib:
    def __init__(self):
        csv_handler = CSVHandler('Dataset\\min-max-normalization.csv')
        self.dataframe = csv_handler.read_csv()

    def Normalization(self):
        self.dataframe.replace('?', pd.NA, inplace=True)
        num_cols = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15']
        self.dataframe[num_cols] = self.dataframe[num_cols].apply(pd.to_numeric, errors='coerce')
        self.dataframe[num_cols] = self.dataframe[num_cols].fillna(self.dataframe[num_cols].mean())

        for col in num_cols:
            min_val = self.dataframe[col].min()
            max_val = self.dataframe[col].max()
            self.dataframe[col] = self.dataframe[col].apply(lambda x: (x - min_val) / (max_val - min_val) * 100)

        # Thực hiện xóa các giá trị trùng lặp trong mỗi cột
        for col in self.dataframe.columns:
            self.dataframe[col] = self.dataframe[col].where(~self.dataframe[col].duplicated(), np.nan)

        # Dịch chuyển các dữ liệu khác NaN lên đầu cột
        for col in self.dataframe.columns:
            non_na = self.dataframe[col].dropna()
            na_values = self.dataframe[col][self.dataframe[col].isna()]
            self.dataframe[col] = pd.concat([non_na, na_values], axis=0, ignore_index=True)

        path = 'Dataset\\predict\\min-max-normalization-nolib.csv'
        csv_handler = CSVHandler(path)
        csv_handler.write_csv(self.dataframe, path)
        os.startfile(path)
