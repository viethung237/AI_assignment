import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.01, regLambda=0.01, epsilon=0.0001, maxNumIters=10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost
        '''
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        J = -(1/m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
        regularization_term = (regLambda / (2 * m)) * np.sum(theta[1:]**2)
        J += regularization_term
        return J.item()

    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n, d = X.shape
        theta = theta.reshape(-1, 1)
        y = y.reshape(-1, 1)
        h = self.sigmoid(X @ theta)

        gradient = (1/n) * X.T @ (h - y)

        # Regularization term (skip the bias term)
        regularization_term = (regLambda / n) * theta[1:]
        
        # Combine the gradients
        gradient[1:] += regularization_term
        
        return gradient


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n, d = X.shape
        X = np.column_stack((np.ones((n, 1)), X))  # Add a column of ones for the bias term
        self.theta = np.zeros((d + 1, 1))

        for _ in range(self.maxNumIters):
            gradient = self.computeGradient(self.theta, X, y, self.regLambda)
            self.theta -= self.alpha * gradient.reshape(-1, 1)

        print("Theta after fit:", self.theta)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        # Add a bias term
        X_bias = np.column_stack([np.ones(X.shape[0]), X])

        # Compute probabilities using the sigmoid function
        probabilities = self.sigmoid(X_bias @ self.theta)

        # Convert probabilities to binary predictions (0 or 1)
        predictions = (probabilities >= 0.5).astype(int)

        return predictions


    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1 / (1 + np.exp(-Z))
