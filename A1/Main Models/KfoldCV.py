import numpy as np
import random
import math

def KCV(X,y, k):
    N,D = X.shape
    fold_size = math.floor(N / k)
    merge = np.concatenate([X,y])
    np.shuffle(merge)








