from utils.CSVHandler import CSVHandler
from LinearRegression.LinearRegressionNoLib import LinearRegressionNoLib as LR0Lib

def main():
    csv_handler = CSVHandler('Dataset\linear-regression.csv')
    dataframe = csv_handler.read_csv()
    print(dataframe)
    print('--------------------------------')

    lr0Lib = LR0Lib(dataframe)

if __name__ == "__main__":
    main()