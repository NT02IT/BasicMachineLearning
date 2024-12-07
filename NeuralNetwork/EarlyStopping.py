class EarlyStopping:
    def __init__(self, patience=5, delta=0.00001):
        self.patience = patience  # Number of epochs with no improvement
        self.delta = delta  # Minimum change to qualify as an improvement
        self.best_loss = None  # Best loss so far
        self.counter = 0  # Counter to track epochs with no improvement
        self.early_stop = False  # Flag to indicate if early stopping should occur
    
    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0  # Reset counter if there's an improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # Trigger early stopping