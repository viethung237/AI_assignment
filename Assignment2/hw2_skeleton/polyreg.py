import numpy as np
from sklearn.preprocessing import StandardScaler

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:
    def __init__(self, degree=1, regLambda=1E-8, alpha=0.01, max_iter=1000):
        '''
        Constructor
        '''
        self.degree = degree
        self.regLambda = regLambda
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.zeros((degree + 1, 1))  # Initialize theta as a column vector of zeros
        self.scaler = StandardScaler()

    def polyfeatures(self, X, degree):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        X_poly = np.power(X, np.arange(1, degree + 1))
        return X_poly

    def gradient_descent(self, X, y):
        m = len(y)
        for _ in range(self.max_iter):
            # Compute predictions and errors
            h = np.dot(X, self.theta)
            error = h - y

            # Compute gradient
            grad = (1/m) * X.T @ error

            # Update theta
            self.theta -= self.alpha * grad

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        X_poly = self.polyfeatures(X, self.degree)

        # Apply feature scaling
        X_scaled = self.scaler.fit_transform(X_poly)

        # Add bias term
        X_scaled = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

        print("Shapes before gradient descent:")
        print("X_scaled.shape:", X_scaled.shape)
        print("y.shape:", y.shape)

        # Ensure Xtrain_subset and Ytrain_subset are 2D arrays
        X_scaled = X_scaled.reshape(-1, self.degree + 1)
        y = y.reshape(-1, 1)

        # Use gradient descent for regularized linear regression
        self.gradient_descent(X_scaled, y)

        print("Shapes after gradient descent:")
        print("self.theta.shape:", self.theta.shape)


    def predict(self, X):
        X_poly = self.polyfeatures(X, self.degree)

        if len(X_poly.shape) == 1:
            X_poly = X_poly.reshape(-1, 1)  # Ensure X_poly is a 2D array

        X_scaled = self.scaler.transform(X_poly)

        # Add bias term
        X_scaled = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

        return X_scaled @ self.theta




#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------

def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in range(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]

        print("Shapes before fit:")
        print("Xtrain_subset.shape:", Xtrain_subset.shape)
        print("Ytrain_subset.shape:", Ytrain_subset.shape)

        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset, Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset
        errorTrain[i] = np.multiply(err, err).mean()
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest
        errorTest[i] = np.multiply(err, err).mean()
    
    return (errorTrain, errorTest)
