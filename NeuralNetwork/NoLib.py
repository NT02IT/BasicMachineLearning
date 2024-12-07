import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from Normalization.Normalization import Normalization


class BinaryClassificationNN:
    def __init__(self, path, train_size=0.5):
        # Load and preprocess data
        self.df = pd.read_csv(path)
        cols = ['person_gender', 'person_education', 'person_home_ownership', 'previous_loan_defaults_on_file', 'loan_intent']
        for col in cols:
            self.df[col] = self.df[col].astype('category').cat.codes
        print(self.df.head(10))
        self.df = self.minMaxNormalizationWlib(self.df, 0, 1)
        print(self.df.head(10))
        self.df = self.df.iloc[1: , :]
        shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.X = shuffled_df.iloc[:, :-1].values  # Input features
        self.y = shuffled_df.iloc[:, -1].values  # Labels (binary: 0 or 1)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size=train_size, random_state=42)

    def loss(self, y, y_hat):
        m = y.shape[0]
        c = 0
        y_hat_flat = y_hat.flatten()
        for i in range(0, m):
            c = c + (y[i] - y_hat_flat[i]) ** 2
        c = (1/m) * c
        return c

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def initialize_parameters(self, input_dim, hidden_dim):
        np.random.seed(0)
        W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        b1 = np.zeros((1, hidden_dim))
        W2 = np.random.randn(hidden_dim, 1) * 0.01  # Output dimension is 1
        b2 = np.zeros((1, 1))
        self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def forward_prop(self, X):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        z1 = X.dot(W1) + b1
        # print(z1)
        a1 = self.sigmoid(z1)
        z2 = a1.dot(W2) + b2
        a2 = self.sigmoid(z2)  # Output layer uses sigmoid
        cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
        return cache

    def backward_prop(self, X, y, cache):
        m = X.shape[0]
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        a1, a2 = cache['a1'], cache['a2']

        dz2 = a2 - y.reshape(-1, 1)  # Gradient of loss w.r.t. output
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0)

        dz1 = dz2.dot(W2.T) * self.sigmoid_derivative(a1)
        dW1 = X.T.dot(dz1)
        db1 = np.sum(dz1, axis=0)

        grads = {'dW1': dW1 / m, 'db1': db1 / m, 'dW2': dW2 / m, 'db2': db2 / m}
        return grads

    def update_parameters(self, grads, learning_rate):
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        
        # Update parameters
        W1 -= learning_rate * grads['dW1']
        b1 -= learning_rate * grads['db1']
        W2 -= learning_rate * grads['dW2']
        b2 -= learning_rate * grads['db2']
        
        # Store and return parameters
        self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def predict(self, X):
        cache = self.forward_prop(X)
        y_hat = cache['a2']
        return (y_hat >= 0.5).astype(int)  # Convert probabilities to binary labels

    def calc_accuracy(self, X, y):
        predictions = self.predict(X)
        m = y.shape[0]
        c = 0
        y_hat_flat = predictions.flatten()
        for i in range(0, m):
            c = c + np.sum(y[i] == y_hat_flat[i])
        c = (100/m) * c
        # print(c)
        return c
    
    def minMaxNormalizationWlib(self, df, new_min, new_max):
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df_normalized = df.copy()
                scaler = MinMaxScaler(feature_range=(new_min, new_max))        
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df_normalized[col] = scaler.fit_transform(df[[col]])        
        return df_normalized
    
    def evaluate_model(self):
        # Make predictions
        y_pred = self.predict(self.X_test)

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
        print(f"Final Loss: {self.losses[-1]}")

    def train(self, X, y, hidden_dim, epochs, learning_rate, patience=10):
        self.initialize_parameters(X.shape[1], hidden_dim)
        self.losses = []
        accuracy = []
        early_stopping = EarlyStopping.EarlyStopping(patience=patience)
        for epoch in range(epochs):
            cache = self.forward_prop(X)
            loss = self.loss(y, cache['a2'])
            self.losses.append(loss)
            
            early_stopping(loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}/{epochs}, Loss: {loss:.4f}")
                break

            grads = self.backward_prop(X, y, cache)
            self.update_parameters(grads, learning_rate)
            
            if epoch % 50 == 0:
                accu = self.calc_accuracy(X, y)
                accuracy.append(accu)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss: .4f}, Accuracy: {self.calc_accuracy(X, y):.2f}%")
        self.evaluate_model()
        figure, axis = plt.subplots(1, 2)
        axis[0].plot(self.losses)
        axis[0].set_title("Training Loss")
        # axis[1].plot(accuracy)
        axis[1].plot([i * 50 for i in range(len(accuracy))], accuracy)
        axis[1].set_title('Training Accuracy')
        plt.show()
        
# Load dataset and train the model
nn = BinaryClassificationNN('datasets\\neural-network\\loan_data.csv')
nn.train(X=nn.X_train, y=nn.y_train, hidden_dim=5, epochs=4500, learning_rate=0.5)
