
#Making predictions with our calculated weights
import numpy as np
import math
# import mglearn
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Logistic_Regression():

    def __init__(self, penalty=None, max_iter=None, random_state=None, lr=None, lambdaa=None, coef = None, beta=None):
        self.lambdaa = lambdaa
        self.coef = coef #after we fit the model, self.coef will contain the estimated weights
        self.max_iter = max_iter # maximum number of iterations to run gradient descent
        self.penalty = penalty #Penalty is to indicate whether we will use L2 regularization
        self.random_state = random_state #we fix a random seed so we achieve the same 'randomized' results
        self.lr = lr  #learning rate of gradient descent
        self.beta = beta


    ##SIGMOID IS OUR PRED() FUNCTION!
    def sigmoid(self,X, weights): #takes in the design matrix and weights and spits out a vector of calculated values
        X = np.array(X) #convert to np array
        pred = np.zeros(X.shape[1])
        logit = np.dot(X, weights)
        pred = (1 / (1 + np.exp(-logit)))
        #print('sigmoid value:{%.2f} and predicted value: {}'.format(pred, np.round(pred)))
        return pred


    def gradient(self,X, y, weights):
         N, D = X.shape
         yhat = self.sigmoid(X,weights)
         grad = (np.dot(X.T, yhat - y))
         if self.penalty == 'l1':
             grad[1:] += self.lambdaa * np.sign(weights[1:])
         return grad


    def GradientDescent(self, X, y, lr, max_iter, eps = 1e-7):
        lambdaa = self.lambdaa
        iter = 0
        N,D = X.shape
        weights = np.zeros(D)
        g = np.inf
        beta = self.beta
        counter = 1
        dw = 1
        while iter < max_iter and np.linalg.norm(g) > eps:
            #print('iter:{}'.format(iter))
            #self.lr = self.lr / (iter + 1)
            #self.lr = (1/iter**0.51)
            g = self.gradient(X,y, weights)
            dw = (1 - beta) * g + beta * dw #implementation of exponential moving average
            weights = weights - self.lr * dw
            #weights = weights - self.lr*g   #changed lr -> self.lr
            #print('iter: {} \t weights: {}'.format(iter, weights))
            iter+=1
        return weights

    #OUR FIT FUNCTION
    def fit(self, X, y, sample_weight = False):  #X is our training design matrix, y is corresponding target vector.
        coef = self.GradientDescent(X, y,  self.lr , self.max_iter)
        self.coef = coef
        return coef                                              # Sample weight is array of weights that are assigned to individual samples. If not provided, then each sample is given unit weight.

    def score(self, X,y):
        total = X.shape[0]
        correct = 0.0
        weights = self.coef   #use sigmoid function to predict the value (i.e if pred = 0.6, we predict y is in class 1)
        pred = np.round(self.sigmoid(X, weights))
        for i in range(X.shape[0]): #iterate over all instances
            if(pred[i] == y[i]):
                correct += 1   #if our prediction is correct we add to our count
    #changed score to return float
        return float(correct/total)


    def KfoldCV(self, X, y, k):
        np.random.seed(self.random_state)  #setting random seed allows us to replicate results
        N,D = X.shape
        shuffle_sequence = np.random.permutation(X.shape[0])
        X_shuffled = X[shuffle_sequence]
        y_shuffled = y[shuffle_sequence]
        train_size = math.floor(N/k)
        results = []
        results_train=[]
        counter = 0
        for i in range(k):
            X_temp = np.copy(X_shuffled)
            y_temp = np.copy(y_shuffled)
            X_test = X_shuffled[counter:counter+train_size, :]
            y_test = y_shuffled[counter:counter+train_size]
            for j in range(counter,counter+train_size):
                np.delete(X_temp, j)
                np.delete(y_temp,j)


            self.fit(X_temp, y_temp)
            results.append(self.score(X_test, y_test))
            results_train.append(self.score(X_temp,y_temp))
            counter += train_size
        #print("Hello : {}".format(results))
        return (results,results_train)


#optimal parameters for ionosphere with momentum/l1 regularization:
# beta = 0.9, lr = 3e-5, lambda = 0.5, iteration = 10,000
#vs beta = 0.5

#optimal parameters for breast cancer w momentum/l1 regularizatoin:
#beta = 0.9, lr = 3e-5, lambda = 0.5, iteration = 400

#Optimal parameters for banknote w momentum/l1:
#beta = 0.9 lr = 3e-5, lambda = 0.5, iteration 8000