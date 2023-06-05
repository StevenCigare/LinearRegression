from numpy import linalg


class LinearRegression():
    def __init__(self):
        #matrice of coefficients
        self.coefficients = None

    def train(self, X, y):
        X_T = X.T #transpone 
        self.coefficients = linalg.inv(X_T @ X) @ X_T @ y 
        self.coefficients.index = X.columns
        return self.coefficients
    
    def predict(self, X):
        #multiplication of test values by coefficients give us the predicted value 
        return X @ self.coefficients
    
    def SSR(self, predictions, y):
        #sum of squared residuals, the sum of squared differences between the actual and predicted values 
        return ((predictions - y) ** 2).sum()
    
    def SST(self, y):
        #sum of squares total, the sum of squared differences between the actual value and the mean value
        return ((y - y.mean()) ** 2).sum()
    
    def r_squared(self, predictions, y):
        #how well is the model aligned
        return 1 - self.SSR(predictions, y)/self.SST(y)
    
        