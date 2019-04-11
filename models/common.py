## common
import numpy as np

def prep(X,Y):
    X_tr_left = []
    X_tr_right = []
    Y_tr = []
    for x,y in zip(X, Y):
        x1,x2 = x
        Y_tr.append(y)
        X_tr_left.append(x1)
        X_tr_right.append(x2)
        
    return np.asarray(X_tr_left).reshape((-1,1)),np.asarray(X_tr_right).reshape((-1,1)),np.asarray(Y_tr).reshape((-1,1))

def read_data(filename):
    out = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('|')
            out.append(tuple(l))
    return out


