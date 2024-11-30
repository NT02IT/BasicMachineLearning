from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


class Normalization:    
    # ENCODE DATA Chuyển từ chữ sang số
    @staticmethod
    def encode_dataframe(df):
        label_encoders = {} # Từ điển lưu trữ LabelEncoder cho mỗi cột

        for col in df.columns:
            if df[col].dtype == 'object':  # Kiểm tra nếu cột là kiểu chuỗi
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col])  # Mã hóa cột chuỗi thành số
                label_encoders[col] = encoder  # Lưu trữ encoder cho cột vào từ điển
        
        return df, label_encoders
    
    @staticmethod
    def encode_dataframe_with_encoders(df, label_encoders):
        for col in df.columns:
            if col in label_encoders:  # Kiểm tra xem cột có trong danh sách LabelEncoder không
                encoder = label_encoders[col]
                if df[col].dtype == 'object':  # Nếu cột là kiểu chuỗi
                    df[col] = encoder.transform(df[col])  # Áp dụng transform
        return df

    
    # DECODE DATA Chuyển từ số về lại chữ ban đầu
    @staticmethod
    def decode_dataframe(df, label_encoders):
        for col in df.columns:
            if col in label_encoders:  # Kiểm tra nếu cột đã có LabelEncoder
                encoder = label_encoders[col]
                df[col] = encoder.inverse_transform(df[col])  # Giải mã các giá trị số thành chuỗi        
        return df
    
    # MinMax Normalization With Library
    @staticmethod
    def minMaxNormalizationWlib(df, new_min, new_max):
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df_normalized = df.copy()
                scaler = MinMaxScaler(feature_range=(new_min, new_max))        
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df_normalized[col] = scaler.fit_transform(df[[col]])        
        return df_normalized
    
    # MinMax Normalization No Library
    @staticmethod
    def minMaxNormalizationNolib(df, new_min, new_max):
        df_normalized = df.copy()
        
        for col in df_normalized.columns:
            if df_normalized[col].dtype in ['int64', 'float64']:
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                
                if max_val - min_val == 0:
                    df_normalized[col] = new_min
                else:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val) * (new_max - new_min) + new_min
        
        return df_normalized