import random
import numpy as np
from math import sqrt, acos, degrees
import math
import matplotlib.pyplot as plt

'''python versin 3.5.0'''
'''Yiqing Guo'''


def angle(d1, d2):
    top = np.dot(d1.T, d2)
    bot = np.linalg.norm(d1)*np.linalg.norm(d2)
    return round(degrees(acos(top/bot)), 2)

def randomgenrate(n, d):
    X = np.zeros(n)
    choseset = [-1, 1]
    for i in range (0, n):
        d1 = np.zeros(d)
        d2 = np.zeros(d)
        for v in range(0, d):
            d1[v] = random.choice(choseset)
            d2[v] = random.choice(choseset)
        X[i] = angle(d1, d2)
        #print (d1, d2)
    return X

def EPMS(X):
    n = X.shape[0]
    angleset = np.unique(X)
    size = angleset.shape[0]
    EPSM_set = np.zeros([size, 2])
    for i in range(0, size):
        EPSM_set[i,0] = angleset[i]
    for i in range(0, n):
        for v in range (0, size):
            if X[i] == EPSM_set[v, 0]:
                EPSM_set[v, 1]+=1
    for i in range(0, size):
        EPSM_set[i,1] =round( EPSM_set[i,1]/n, 2 )
    return EPSM_set

def figer_plot(EPSM_set, title):
    row = EPSM_set.shape[0]
    width = 5
    plt.figure();
    plt.bar(EPSM_set[0:row, 0], EPSM_set[0:row, 1], color="blue")
    plt.ylabel('Probability')
    plt.xlabel('Degree')
    plt.title(title)
    
def feature(X):
    print('Max:', max(X))
    print('Min:', min(X))
    print ('Range value:' , max(X) - min(X))
    print('Mean:', np.mean(X))
    print('Variance:', np.var(X))    
    
        
if __name__ == '__main__':
    np.set_printoptions(suppress=True)  
    X = randomgenrate(100000, 10)
    feature (X)
    title = 'n= 10'
    figer_plot(EPMS(X), title)
    X = randomgenrate(100000, 100)
    feature (X)
    title = 'n= 100'
    figer_plot(EPMS(X), title)
    X = randomgenrate(100000, 1000)
    feature (X)
    title = 'n= 1000'
    figer_plot(EPMS(X), title)    