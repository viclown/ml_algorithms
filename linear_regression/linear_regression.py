import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
        
    def fit(self, X: np.array, y: np.array):
        new_x = np.hstack((X, np.ones((X.shape[0], 1))))
        self.weights = np.linalg.pinv(new_x).dot(y)
        
    def predict(self, X: np.array) -> np.array:
        new_x = np.hstack((X, np.ones((X.shape[0], 1))))
        return new_x.dot(self.weights)
    
    def r_score(self, y_true: np.array, y_pred: np.array) -> float:
        u = ((y_true - y_pred)**2).sum()
        v = ((y_true - y_true.mean())**2).sum()
        r_squared = 1 - u/v
        return r_squared
    
    def coefficients(self) -> np.array:
        return self.weights
    
    def metrics_mean_error(self, y_true: np.array, y_pred: np.array) -> float:
        return (np.fabs(y_true - y_pred)).sum()/(y_true.shape[0])