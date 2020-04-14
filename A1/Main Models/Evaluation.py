import math
import Naive_Bayes
from Naive_Bayes import *
import numpy as np
import Logistic_Regression
from Logistic_Regression import *
import pandas as pd
import timeit
from timeit import *

class Evaluation():
    def __init__(self, accuracies_3_1 = None, accuracies_3_3 = None, train_acc_3_3 = None, runtimes = None, binary_threshold = 0):
        # stores the accuracies and runtimes of the algorithms on the 4 datasets with indices:
        # 0: Adult (continuous and binary)
        # 1: Banknote (all continuous)
        # 2: Ionosphere (all continuous)
        # 3: Breast cancer (all binary)
        self.accuracies_3_1 = accuracies_3_1
        self.accuracies_3_3 = accuracies_3_3
        self.train_acc_3_3 = train_acc_3_3
        self.runtimes = runtimes
        # 0 by default (i.e. defaults to handling continuous features in naive bayes)
        self.binary_threshold = binary_threshold
        self.NB = Naive_Bayes()
        self.LR = Logistic_Regression()

    def fit_and_predict(self, xTrain, yTrain, xTest, yTest, test_type):
        if test_type == "lr":
            self.LR.fit(xTrain, yTrain)
            return self.LR.score(xTrain, yTrain)
        else:
            self.NB.get_p_features(xTrain, yTrain)
            self.NB.get_priors(yTrain)
            return self.NB.score(xTest, yTest)


    def K_fold_CV(self, X, y, k):
        #np.random.seed(45)  #setting random seed allows us to replicate results
        N,D = X.shape
        shuffle_sequence = np.random.permutation(X.shape[0])
        X_shuffled = X[shuffle_sequence, :]
        y_shuffled = y[shuffle_sequence]
        train_size = math.floor(N/k)
        results = np.zeros((2, k))
        counter = 0
        for i in range(k):
            X_test = X_shuffled[counter:counter+train_size, :]
            y_test = y_shuffled[counter:counter+train_size]

            X_train = np.concatenate([X_shuffled[0:counter, :], X_shuffled[counter+train_size:N, :]])
            y_train = np.concatenate([y_shuffled[0:counter], y_shuffled[counter+train_size:N]])
            results[0, i] = (self.fit_and_predict(X_train, y_train, X_test, y_test, "lr"))
            results[1, i] = (self.fit_and_predict(X_train, y_train, X_test, y_test, "nb"))
            counter += train_size

        # returns the average accuracy over k folds
        return np.average(results, axis=1)

    # e.g. file_name = "Adult_Scaled.csv"
    def separate_X_Y(self, file_name):
        df = pd.read_csv(file_name)
        if (file_name == "ionosphere90.csv" or file_name == "ionosphere10.csv" or file_name == "adult_unscaled90.csv" or file_name == "adult_unscaled10.csv"):
            return np.array(df.iloc[:, 2:-1]), np.array(df.iloc[:, -1])
        return np.array(df.iloc[:, 1:-1]), np.array(df.iloc[:, -1])

    def main(self):
        files90 = ["adult_unscaled90.csv", "banknote90.csv", "ionosphere90.csv", "Breast90.csv"]
        files10 = ["adult_unscaled10.csv", "banknote10.csv", "ionosphere10.csv", "Breast10.csv"]
        acc_3_1 = np.zeros((2, 4))
        acc_3_3 = np.zeros((2, 4, 4))
        train_acc_3_3 = np.zeros((2, 4, 4))

        # task 3-1
        for i in range(0,len(files90)):
            print("Starting to process file " + files90[i])
            X, Y = self.separate_X_Y(files90[i])
            X_10, Y_10 = self.separate_X_Y(files10[i])
            # set binary thresholds and hyperparameters
            # optimal parameters for ionosphere with momentum/l1 regularization:
            # vs beta = 0.5

            # optimal parameters for breast cancer w momentum/l1 regularizatoin:

            # Optimal parameters for banknote w momentum/l1:
            if i == 0:
                self.LR = Logistic_Regression(lr=0.006, penalty='none', max_iter=1750, random_state=0, lambdaa=0.0, beta = 0.9)
                self.NB.set_bin_thresh(79)
            elif i == 1:
                # all features are continuous
                self.LR = Logistic_Regression(lr=3e-5, penalty='l1', max_iter=8000, random_state=0, lambdaa=0.5, beta=0.9)
                self.NB.set_bin_thresh(0)
            elif i == 2:
                # all features are continuous
                self.LR = Logistic_Regression(lr=3e-5, penalty='l1', max_iter=10000, random_state=0, lambdaa=0.5, beta=0.9)
                self.NB.set_bin_thresh(0)
            else:
                # all features are binary
                self.LR = Logistic_Regression(lr=3e-5, penalty='l1', max_iter=600, random_state=0, lambdaa=0.75, beta=0.9)
                self.NB.set_bin_thresh(len(X[0]))
            acc_3_1[:, i] = self.K_fold_CV(X, Y, 5)

            N, D = X.shape
            shuffle_sequence = np.random.permutation(X.shape[0])
            X_shuffled = X[shuffle_sequence, :]
            Y_shuffled = Y[shuffle_sequence]

            # train model and do CV on 100%, 85%, 70%, 65% of the other data points
            percent_train = [1, 0.85, 0.7, 0.65]
            for j in range(4):
                print("Running " + str(percent_train[j]) + " " + files90[i])
                X_train = X_shuffled[0:math.floor(N*percent_train[j]), :]
                Y_train = Y_shuffled[0:math.floor(N*percent_train[j])]
                train_acc_3_3[:, i, j] = self.K_fold_CV(X_train, Y_train, 5)
                acc_3_3[0, i, j] = self.LR.score(X_10, Y_10)
                acc_3_3[1, i, j] = self.NB.score(X_10, Y_10)

        self.accuracies_3_1 = acc_3_1
        self.accuracies_3_3 = acc_3_3
        self.train_acc_3_3 = train_acc_3_3

        return acc_3_1, acc_3_3

run = Evaluation()
run.main()
print(run.accuracies_3_1)
print(run.accuracies_3_3)
print(run.train_acc_3_3)
# run.tim()


# files90 = ["adult_unscaled90.csv", "banknote90.csv", "ionosphere90.csv", "Breast90.csv"]
# files10 = ["adult_unscaled10.csv", "banknote10.csv", "ionosphere10.csv", "Breast10.csv"]
# for i in range(4):
#     df = pd.read_csv(files90[i])
#     df10 = pd.read_csv(files10[i])
#     X, D = df.shape
#     X1, D1 = df10.shape
#     print(X, X1)
#     print(D)

# files90 = ["adult_unscaled90.csv"]
# df = pd.read_csv(files90[0])
# X = np.array(df.iloc[:, 2:-1])
# Y = np.array(df.iloc[:, -1])
#
# LR = Logistic_Regression(lr=3e-5, penalty='l1', max_iter=10000, random_state=0, lambdaa=0.5, beta=0.9)
# print(X.shape)
# print(LR.fit(X, Y))

