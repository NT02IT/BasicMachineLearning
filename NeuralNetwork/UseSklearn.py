import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

class UseSklearn:
    def __init__(self, csv_path):
        # Load the CSV file
        self.df = pd.read_csv(csv_path)
        self.preprocess_data()

    def preprocess_data(self):
        # Convert categorical columns to numerical codes
        cols = ['person_gender', 'person_education', 'person_home_ownership', 
                'previous_loan_defaults_on_file', 'loan_intent']
        for col in cols:
            self.df[col] = self.df[col].astype('category').cat.codes
        
        # Normalize the data
        self.df = self.min_max_normalization(self.df, 0, 1)
        
        # Separate features (X) and target (y)
        self.X = self.df.drop('loan_status', axis=1).values
        self.y = self.df['loan_status'].values
        
        # Split into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)

    def min_max_normalization(self, df, new_min, new_max):
        df_normalized = df.copy()
        scaler = MinMaxScaler(feature_range=(new_min, new_max))
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df_normalized[col] = scaler.fit_transform(df[[col]])
        return df_normalized

    def train_model(self):
        # Create the MLPClassifier
        self.mlp = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, learning_rate_init=0.01, random_state=42)

        # Train the model
        self.mlp.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Make predictions
        y_pred = self.mlp.predict(self.X_test)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy:.4f}')

        # Print classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Print confusion matrix
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        # Print loss value
        print(f"Final Loss: {self.mlp.loss_}")

    def plot_loss_curve(self):
        # Optional: Plotting the loss curve
        plt.plot(self.mlp.loss_curve_)
        plt.title('Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize the class with the path to the CSV file
    sklearn_model = UseSklearn('datasets\\neural-network\\loan_data.csv')

    # Train the model
    sklearn_model.train_model()

    # Evaluate the model
    sklearn_model.evaluate_model()

    # Plot the loss curve
    sklearn_model.plot_loss_curve()