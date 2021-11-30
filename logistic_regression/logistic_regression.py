import numpy as np

class LogisticRegression:
    def __init__(self):
        self.weights: np.ndarray = None
        self.bias = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate: float = 0.001, epochs: int = 100):
        self.weights = np.zeros((X.shape[1], 1))
        
        for epoch in range(epochs):
            derivative_w = np.zeros((X.shape[1], 1))
            derivative_b = 0
                
            z_pred = np.array(X.dot(self.weights) + self.bias, dtype=np.float32)
            y_pred = 1 / (1 + np.exp(-z_pred))
            
            derivative_w = np.transpose(X).dot(y_pred - y) / X.shape[1]
            derivative_b = np.sum(y_pred - y) / X.shape[1]
            
            self.weights = self.weights - learning_rate * derivative_w
            self.bias = self.bias - learning_rate * derivative_b
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z_pred = X.dot(self.weights) + self.bias
        z_pred = z_pred.astype(float)
        y_pred_proba = 1 / (1 + np.exp(-z_pred))
            
        return y_pred_proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        y_pred = self.predict_proba(X) >= threshold
            
        return y_pred.astype(int)
    
    def confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        true_negative = 0; true_positive = 0; false_positive = 0; false_negative = 0;

        for i in range(len(y_true)): 
            if y_pred[i] == 1 and y_true[i] == 1:
                true_positive += 1
            elif y_pred[i] == 1 and y_true[i] == 0:
                false_positive += 1
            elif y_pred[i] == 0 and y_true[i] == 0:
                true_negative +=1
            elif y_pred[i] == 0 and y_true[i] == 1:
                false_negative += 1
        
        return np.array([[true_negative, false_positive], [false_negative, true_positive]])
    
    def precision_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        confusion_matrix = self.confusion_matrix(y_true, y_pred)
        true_positive = confusion_matrix[1][1]
        false_positive = confusion_matrix[0][1]
        
        return true_positive / (true_positive + false_positive)
    
    def recall_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        confusion_matrix = self.confusion_matrix(y_true, y_pred)
        true_positive = confusion_matrix[1][1]
        false_negative = confusion_matrix[1][0]
        
        return true_positive / (true_positive + false_negative)
    
    def f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        recall = self.recall_score(y_true, y_pred)
        precision = self.precision_score(y_true, y_pred)
        
        return 2 * (precision * recall) / (precision + recall)