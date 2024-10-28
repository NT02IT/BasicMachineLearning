import pandas as pd
import numpy as np

class ValidateNoLib:
    def __init__(self, ytrue, ypred):
        # Giả sử ytrue là DataFrame, chọn cột đầu tiên
        self.ytrue = ytrue.iloc[:, 0] if isinstance(ytrue, pd.DataFrame) else pd.Series(ytrue)
        self.ypred = ypred.iloc[:, 0] if isinstance(ypred, pd.DataFrame) else pd.Series(ypred)

        # Lấy các lớp
        self.classes = np.unique(np.concatenate((self.ytrue, self.ypred)))

    def getSampleSize(self):
        return len(self.ytrue)
    
    def getSampleClasses(self):
        return self.classes

    def confusionMatrix(self):
        self.matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)
        for true_label, pred_label in zip(self.ytrue, self.ypred):
            self.matrix[true_label, pred_label] += 1        
        return self.matrix

    def accuracy(self):
        # Tính độ chính xác
        correct_predictions = (self.ytrue == self.ypred).sum()
        return correct_predictions / self.getSampleSize()

    def precision(self, label):
        # Tính precision cho một nhãn cụ thể
        tp = ((self.ytrue == label) & (self.ypred == label)).sum()
        fp = ((self.ytrue != label) & (self.ypred == label)).sum()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    def recall(self, label):
        # Tính recall cho một nhãn cụ thể
        tp = ((self.ytrue == label) & (self.ypred == label)).sum()
        fn = ((self.ytrue == label) & (self.ypred != label)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    def fscore(self, label, beta=1):
        # Tính F-score cho một nhãn cụ thể
        p = self.precision(label)
        r = self.recall(label)
        return (1 + beta**2) * (p * r) / (beta**2 * p + r) if (beta**2 * p + r) > 0 else 0.0


# Ví dụ sử dụng
ytrue_df = pd.DataFrame({'true_labels': [0, 1, 2, 2, 0, 1, 0, 2, 1, 3]})
ypred = pd.DataFrame({'true_labels': [0, 0, 2, 1, 3, 1, 2, 2, 1, 3]})

validator = Validate(ytrue_df, ypred)

# Xuất các giá trị
print("Số lượng mẫu:", validator.getSampleSize())
print("Các phân lớp:", validator.getSampleClasses())
print("Ma trận nhầm lẫn:\n", validator.confusionMatrix())
print("Độ chính xác:", validator.accuracy())
for label in range(4):  
    print(f"Precision cho lớp {label}:", validator.precision(label))
    print(f"Recall cho lớp {label}:", validator.recall(label))
    print(f"F-score cho lớp {label}:", validator.fscore(label))
