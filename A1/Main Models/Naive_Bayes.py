import numpy as np
from numpy import newaxis
import math
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_validate,cross_val_predict,cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB

class Naive_Bayes():

    def __init__(self, p_priors = None, p_binary = None, p_means = None, p_stds = None, binary_threshold = None):
        self.p_priors = p_priors # after we fit the model, will contain the prior probabilities (p(c = 0), p(c = 1))
        self.p_binary = p_binary # after we fit the model, will contain the weights of binary features
        self.p_means = p_means # after we fit the model, will contain the means of the gaussians
        self.p_stds = p_stds # after we fit the model, will contain the standard deviations of the gaussians
        self.binary_threshold = binary_threshold # contains the index of the column marking boundary btwn binary and continuous features

    def set_bin_thresh(self, t):
        self.binary_threshold = t
    # X is a matrix of the data (instances x features)
    # Y is a binary vector indicating the label
    # returns a vector: [p(c=0), p(c=1)]
    def get_priors(self, Y):
        priors= np.zeros(2)
        #len(Y) is the number of rows in Y
        priors[0] = (len(Y)-np.sum(Y))/len(Y)
        priors[1] = np.sum(Y)/len(Y)
        self.p_priors = priors
        return priors

    # calculates the weights for binary (p_binary) and continuous features (p_means and p_stds)
    def get_p_features(self, X, Y):
        N, D = X.shape
        # p has dimensions: D x 2
        p_binary = np.zeros((2, self.binary_threshold))

        # np.sum(Y) should add up all the 1's in Y, i.e. gives the number of instances where c=1

        # binary features
        # for y = 0
        inds0 = np.where(Y == 0)[0]
        p_binary[0, :] = np.sum(X[inds0, 0:self.binary_threshold], 0) / len(Y[inds0])
        self.p_binary = p_binary

        # for y = 1
        inds1 = np.nonzero(Y)
        p_binary[1, :] = np.sum(X[inds1, 0:self.binary_threshold], 1)/len(Y[inds1])


        # continuous features
        # for y = 0
        p_means = np.zeros((2, D - self.binary_threshold))
        p_stds = np.zeros((2, D - self.binary_threshold))
        p_means[1, :] = np.mean(X[inds1, self.binary_threshold:D], 1)
        p_stds[1, :] = np.std(X[inds1, self.binary_threshold:D], 1)

        # for y = 1
        p_means[0, :] = np.mean(X[inds0, self.binary_threshold:D], 0)
        p_stds[0, :] = np.std(X[inds0, self.binary_threshold:D], 0)
        self.p_means = p_means
        self.p_stds = p_stds
        return (p_binary, p_means, p_stds)

    # returns p_gaussians
    def gaussian(self, X):
        N,D = X.shape
        means = np.array(self.p_means)
        stds = np.array(self.p_stds)
        p_gaussian = np.zeros((N, D-self.binary_threshold, 2))
        for i in range(N):
            row = X[i]
            p_gaussian[i, :, :] = np.transpose(1/(np.sqrt(2*math.pi*stds))*np.exp(-np.power((row[self.binary_threshold:D] - means), 2)/(2*stds)))
        return p_gaussian

    # code something to score the performance (e.g. f1, numcorrect/totalnum)
    def score(self, X, Y):
        YHat = self.predict(X)
        # evaluate performance

        #numcorrect/totalnum
        correct = YHat == Y
        num_correct = sum(correct)
        corr_over_total = num_correct/len(Y)

        #some other measures of performance
        return corr_over_total

    # takes input values X, outputs yhats according to weights learned by Naive Bayes
    def predict(self, X):
        X = np.array(X)
        if np.ndim(X) == 1:
            X = X[newaxis, :]
        N, D = X.shape
        yhat = np.zeros(N)

        p_row = np.zeros((N, self.binary_threshold, 2))
        p_X = np.zeros((N, self.binary_threshold, 2))

        p_gaussian = self.gaussian(X)

        # goes through rows of X
        for i in range(N):
            row = np.squeeze(X[i, :])

            non_zero_indices = np.nonzero(row[0:self.binary_threshold])
            p_row[i, non_zero_indices, 0] = self.p_binary[0, non_zero_indices]
            p_row[i, non_zero_indices, 1] = self.p_binary[1, non_zero_indices]


            zero_indices = np.where(row[0:self.binary_threshold] == 0)[0]
            p_row[i, zero_indices, 0] = 1-self.p_binary[0, zero_indices]
            p_row[i, zero_indices, 1] = 1-self.p_binary[1, zero_indices]

        p_row = np.concatenate([p_row, p_gaussian], 1)

        for i in range(N):
            p_per_class = np.array(self.p_priors)
            for k in range(2):
                for j in range(D):
                    p_per_class[k] = p_per_class[k]*p_row[i, j, k]
            if np.argmax(p_per_class) == 0:
                yhat[i] = 0
            else:
                yhat[i] = 1

        return yhat


# testing
# nb =  Naive_Bayes(binary_threshold=3)
# X = np.array([[0, 0, 1, 3, 2, 1],
#               [0, 1, 0, 9, 7, 8],
#               [1, 1, 1, 10, 6, 9],
#               [0, 0, 1, 1, 1, 2],
#               [0, 0, 1, 0, 3, 1]])
#
# Y = np.array([0, 1, 1, 0, 0])
#
# Z = np.array([0, 1, 1, 9, 7, 9])
# nb.get_priors(Y)
# nb.get_p_features(X, Y)
# print(nb.p_means)
# print(nb.p_stds)
# print(nb.gaussian(X).shape)
# print(nb.gaussian(X))
# print(nb.predict(Z))
# print(nb.score(X, Y))
#
# df_adult = pd.read_csv("Adult_Scaled.csv")
# xg = df_adult.iloc[:, 0:6]
# y = df_adult.iloc[:, -1]
#
# xtrain, xtest, ytrain, ytest = train_test_split(xg, y, random_state=42)
# NBG = GaussianNB()
# NBG.fit(xtrain, ytrain)
# print("Sci-kit gaussian score: " + str(NBG.score(xtest, ytest)))
#
# nbg = Naive_Bayes(binary_threshold=0)
# xtrain = np.array(xtrain)
# ytrain = np.array(ytrain)
# xtest = np.array(xtest)
# ytest = np.array(ytest)
# nbg.get_p_features(xtrain, ytrain)
# nbg.get_priors(ytrain)
# print("our algo gaussian score: " + str(nbg.score(xtest, ytest)))
#
# NBB = BernoulliNB()
# xb = df_adult.iloc[:, 7:-1]
# xtrain, xtest, ytrain, ytest = train_test_split(xb, y, random_state=42)
#
# NBB.fit(xtrain, ytrain)
# print("Sci-kit bernoulli score: " + str(NBB.score(xtest, ytest)))
#
# xtrain = np.array(xtrain)
# ytrain = np.array(ytrain)
# xtest = np.array(xtest)
# ytest = np.array(ytest)
#
# nbb = Naive_Bayes(binary_threshold=len(xtrain[0]))
# nbb.get_p_features(xtrain, ytrain)
# nbb.get_priors(ytrain)
# print("our algo bernoulli score: " + str(nbb.score(xtest, ytest)))


# trying with weights
# df_adult = pd.read_csv("Adult_Unscaled.csv")
# weights = df_adult.iloc[:, 1]
# xg = df_adult.iloc[:, 0:6]
# y = df_adult.iloc[:, -1]
#
# xtrain, xtest, ytrain, ytest = train_test_split(xg, y, random_state=42)
# NBG = GaussianNB()
# NBG.fit(xtrain.iloc[:, [0, 2, 3, 4, 5]], ytrain, xtrain.iloc[:, 1])
# print("Sci-kit gaussian score with weights: " + str(NBG.score(xtest.iloc[:, [0, 2, 3, 4, 5]], ytest)))
#
#
# NBG.fit(xtrain.iloc[:, [0, 2, 3, 4, 5]], ytrain)
# print("Sci-kit gaussian score without weights: " + str(NBG.score(xtest.iloc[:, [0, 2, 3, 4, 5]], ytest)))

