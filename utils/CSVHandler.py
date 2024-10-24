import pandas as pd

class CSVHandler:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_csv(self):
        try:
            df = pd.read_csv(self.file_path)
            return df
        except FileNotFoundError:
            print(f"File {self.file_path} không tồn tại.")
            return pd.DataFrame()  # Trả về DataFrame rỗng nếu không có file.

    def write_csv(self, df):
        df.to_csv(self.file_path, index=False)

    def append_row(self, row):
        try:
            df = pd.read_csv(self.file_path)
            df = df.append(row, ignore_index=True)
        except FileNotFoundError:
            # Nếu file không tồn tại, tạo DataFrame mới với dữ liệu dòng mới.
            df = pd.DataFrame([row])

        # Ghi lại dữ liệu sau khi thêm dòng mới vào file CSV.
        df.to_csv(self.file_path, index=False)
