import pandas as pd
from utils.CSVHandler import CSVHandler
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder # Chuyển đổi danh sách sản phẩm thành định dạng one-hot encoding


class AprioriWLib:
    def __init__(self):
        csv_handler = CSVHandler('datasets\\apriori\\apriori.csv')
        dataframe = csv_handler.read_csv()   
        
        # Chuyển đổi dữ liệu sang dạng transaction
        self.transactions = dataframe.groupby('InvoiceNo')['StockCode'].apply(list).tolist()        

    def training(self):
        encoder = TransactionEncoder()
        onehot = encoder.fit(self.transactions).transform(self.transactions)
        onehot_df = pd.DataFrame(onehot, columns=encoder.columns_)

        # Áp dụng thuật toán Apriori
        frequent_itemsets = apriori(onehot_df, min_support=0.05, use_colnames=True)

        # Tạo quy tắc kết hợp từ các tập hợp sản phẩm
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

        print("Frequent Itemsets:")
        print(frequent_itemsets)
        print("\nAssociation Rules:")
        print(rules)