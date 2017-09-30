import numpy as np
from numpy import linalg
from math import sqrt, acos, degrees
import math
import os
import sys
import matplotlib.pyplot as plt

'''python versin 3.5.0'''
'''Yiqing Guo'''

def readfile(filename):
    x= 0
    #a = np.arange(6)
    for line in open (filename, 'r'):
        item = line.rstrip()
        one = item.split(',')
        one_line= np.array([one[0], one[1],one[2],one[3],one[4],one[5],one[6],one[7],one[8]], dtype=float)
        if(x == 0):
            array = one_line
        if(x!= 0):
            temp = np.vstack((array,one_line)) 
            array = temp
        x += 1
    #print x
    return array

def linearkernel(array):
    (row, col) = array.shape
    kernel = np.zeros((row,row))
    for r in range(0, row):
        for c in range(0, row):
            kernel[r, c] = np.dot(array[r].T, array[c])
    return kernel

def gaussianKernel(array, S):
    (row, col) = array.shape
    kernel = np.zeros((row,row))
    for r in range(0, row):
        for c in range(0, row):
            kernel[r, c] = math.exp(-(np.linalg.norm(array[r]-array[c])**2) / (2*S))
    return kernel

def center(array):
    row= array.shape[0]
    I = np.zeros((row,row))
    for i in range(0, row):
        I[i,i] = 1
    O = np.ones((row,row))
    O = I-O/row
    center = np.dot(O, array)
    center = np.dot(center, O)
    return center

def kernelrun(array):
    (row, col) = array.shape
    orgarray = array
    lineararray = linearkernel(array)
    array = center(lineararray)
    eigVals,eigVects = np.linalg.eigh(array)
    eigValIndice=np.argsort(-eigVals)
    n_eigValIndice=eigValIndice[0:2]
    n_eigVect=eigVects[:,n_eigValIndice]  
    n_eigVals = eigVals[eigValIndice[0:2]]
    #print(eigVals)
    bottom_cov = 0;
    for i in range(0, row):
        if eigVals[i]>0:
            bottom_cov = bottom_cov+eigVals[i]/row
    variance = 0
    for i in range(0, row):
        if eigVals[eigValIndice[i]]>0:
            variance = variance+eigVals[eigValIndice[i]]/row
        rate = variance/bottom_cov
        #print(variance)
        if rate > 0.95:
            print(i+1, 'dimensions are required to capture 95% of the total variance')
            break
    for i in range(0, row):
        for v in range (0,2):
            n_eigVect[i, v] = sqrt(1/n_eigVals[v])*n_eigVect[i, v]
    new_point = np.zeros((row,2))
    for i in range (0,row):
        for v in range (0,2):
            x = 0
            for w in range (0,row):
                x += n_eigVect[i,v]* lineararray[i,w]
            #print(n_eigVect[i,v], x)
            new_point[i,v] = x
    print (new_point)
    return new_point
    
    #print (eigValIndice,eigVals)

def PCA(array):
    (row, col) = array.shape
    mean_array = np.mean(array, axis = 0)
    array = array - mean_array
    cov_array=np.cov(array,rowvar=0)  
    eigVals,eigVects=np.linalg.eig(np.mat(cov_array)) 
    eigValIndice=np.argsort(-eigVals)
    #print(eigVals)
    #print(eigValIndice)
    #print(eigVals,eigVects)
    n_eigValIndice=eigValIndice[0:2]
    n_eigVect=eigVects[:,n_eigValIndice]  
    #print(n_eigVect)
    new_points = array*n_eigVect 
    #print(new_points)
    return new_points

def kernelrunG(array, S):
    (row, col) = array.shape
    orgarray = array
    gaussian_Kernel = gaussianKernel(array, S)
    array = center(gaussian_Kernel)
    eigVals,eigVects = np.linalg.eigh(array)
    eigValIndice=np.argsort(-eigVals)
    n_eigValIndice=eigValIndice[0:2]
    n_eigVect=eigVects[:,n_eigValIndice]  
    n_eigVals = eigVals[eigValIndice[0:2]]
    new_point = np.zeros((row,2))
    for i in range (0,row):
        for v in range (0,2):
            x = 0
            for w in range (0,row):
                x += n_eigVect[i,v]* gaussian_Kernel[i,w]
            #print(n_eigVect[i,v], x)
            new_point[i,v] = x
    return new_point


def plotfigure(plotarray, number, name):
    (row, col) = plotarray.shape
    plt.figure(number)
    plt.plot(plotarray[0:row, 0], plotarray[0:row, 1],'bx') 
    plt.title(name)
    
if __name__ == '__main__':
    filename = 'Concrete_Data.txt'
    a = 2000
    if len(sys.argv) == 3:
            filename = sys.argv[1]
            a = int(sys.argv[2])
    array = readfile(filename)
    #print kernelrun(array)
    pcaarray = PCA(array)
    plotfigure(pcaarray, 1, 'covariance')
    kernel_linear = kernelrun(array)
    plotfigure(kernel_linear, 2, 'linear kernel')
    kernel_gaussian = kernelrunG(array, a)
    plotfigure(kernel_gaussian, 3, 'gaussian kernel')