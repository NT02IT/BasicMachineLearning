import pandas as pd
import numpy as np

class ValidateNoLib:
    def __init__(self, ytrue, ypred):
        # Giả sử ytrue là DataFrame, chọn cột đầu tiên
        self.ytrue = ytrue.iloc[:, 0] if isinstance(ytrue, pd.DataFrame) else pd.Series(ytrue).reset_index(drop=True)
        self.ypred = ypred.iloc[:, 0] if isinstance(ypred, pd.DataFrame) else pd.Series(ypred).reset_index(drop=True)

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
