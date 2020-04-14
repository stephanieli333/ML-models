import re

reviews_train = []
for line in open('/Users/AndrewCheng/Documents/Machine_Learning/movie_data/full_train.txt', 'r'):
    reviews_train.append(line.strip())

reviews_test = []
for line in open('/Users/AndrewCheng/Documents/Machine_Learning/movie_data/full_train.txt', 'r'):
    reviews_test.append(line.strip())