'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO     
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None
        self.mean = None
        self.std = None


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        arr = []
        for i in range(0, X.size):
            cur = []
            for j in range(0, degree):
                cur.append(X[i]**(1 + j))
            arr.append(cur)
        return arr
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        # expand = self.polyfeatures(X,y)
        
        
        # n = len(y)
        # n,d = X.shape
           

        # if type(self.theta) == type(None):
        #     self.theta = np.matrix(np.zeros((d,1)))
        # self.theta = self.gradientDescent(X,y,self.theta)  
        
        expand = self.polyfeatures(X, self.degree)

        expand_np = np.array(expand)

        # get std and mean for training data (to be used for testing as well)
        std = np.std(expand_np, axis=0)
        mean = np.mean(expand_np, axis=0)
        self.std = std
        self.mean = mean

        # standardize data
        expand_np = (expand_np - mean) / std

        # add the zero-th order feature row (i.e. x_0 = 1)
        expand_np = np.c_[np.ones((expand_np.shape[0],1)), expand_np]

        # fit
        n,d = expand_np.shape
        d = d - 1
        regMatrix = self.regLambda * np.eye(d + 1)
        regMatrix[0,0] = 0

        self.theta = np.linalg.pinv(expand_np.T.dot(expand_np) + regMatrix).dot(expand_np.T).dot(y);
        
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        expand = self.polyfeatures(X, self.degree)
        expand_np = np.array(expand)

        # standardize data based on training means and stds
        expand_np = (expand_np - self.mean) / self.std

        # add the zero-th order feature row (i.e. x_0 = 1)
        expand_np = np.c_[np.ones((expand_np.shape[0],1)), expand_np]

        return expand_np.dot(self.theta)


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
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset;
        errorTrain[i] = np.multiply(err, err).mean();
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest;
        errorTest[i] = np.multiply(err, err).mean();
    
    return (errorTrain, errorTest)