'''
Yiqing Guo
CSCI 6390
python version 2.7.11
sample command line input: assign1.py airfoil_self_noise.dat 8 0.0001 out.txt
python file, data file, precision, epsilon, output file
all print text result are save in out.txt
if there are no 5 argument then the output will print in the command windows.
sample command line input with 4 arguments: assign1.py airfoil_self_noise.dat 8 0.0001
if less than 4 argument then the code will using the defualt value.
4 figure are saved as most_correlated.png, anti_correlated.png, least_correlated.png, eigenvectors.png 
'''
import numpy as np
from numpy import linalg as LA
from math import sqrt, acos, degrees
import os
import sys
import matplotlib.pyplot as plt

def readfile(filename):
    x= 0
    #a = np.arange(6)
    for line in open (filename, 'r'):
    #for line in open ('iris.txt', 'r'):
        item = line.rstrip()
        #one = item.split()
        #one_line= np.array([one[0], one[1],one[2],one[3],one[4],one[5]], dtype=float)
        one = item.split(',')
        one_line= np.array([one[0], one[1],one[2],one[3],one[4],one[5],one[6],one[7],one[8]], dtype=float)        
        #one = item.split(',')
        #one_line= np.array([one[0], one[1]], dtype=float)
        if(x == 0):
            array = one_line
        if(x!= 0):
            temp = np.vstack((array,one_line)) 
            array = temp
        x += 1
    #print x
    return array

def sum(array):
    (m, n) = array.shape
    sum_array = np.zeros(n)
    for i in range (0,m):
        for v in range (0,n):
            sum_array[v] = sum_array[v] + array[i, v]
    return sum_array

def transpose(array):
    (m, n) = array.shape
    trans_array = np.zeros((n,m))
    for i in range (0,m):
        for v in range (0,n):  
            trans_array[v, i] = array[i,v]
    return trans_array
               
def multiple(arraya, arrayb):
    (m, n) = arraya.shape
    (p, q) = arrayb.shape
    ans = np.zeros((n,p))
    for i in range (0,n):
        for j in range (0, p):
            sum_item = 0
            for k in range(0, m):
                sum_item += arraya[k,i]*arrayb[j,k]
            ans[i,j] = sum_item
    return ans
    
def mean(array):
    sum_array = sum(array)
    (m, n) = array.shape
    mean = sum_array / m
    return mean

def variance(array):
    mean_array = mean(array)
    x = array - mean_array
    (m, n) = x.shape
    var = np.zeros(n)
    for i in range (0,n):
        sumsq = 0
        for v in range (0, m):
            sumsq += x[v,i]*x[v,i]
        var[i] = sumsq/m
    total_var = var.sum()
    return total_var

def inner(array):
    (m, n) = array.shape
    mean_array = mean(array)
    array = array- mean_array
    transpose_a = transpose(array)
    multi = multiple(array ,transpose_a) 
    ans = multi/m
    return ans

def outter(array):
    (m, n) = array.shape
    mean_array = mean(array)
    array = array- mean_array
    ans = np.zeros((n,n))
    for i in range (0,m):
        temp = np.zeros((n,n))
        for v in range (0,n):
            for q in range (0, n):
                temp[v,q] = array[i,v]*array[i,q]
        ans = ans + temp
    ans = ans/m
    return ans

def Correlation(array):
    (m, n) = array.shape
    array = array- mean_array
    corr_array = []
    for i in range (0, n):
        Z1 = array[0:m, i]
        for q in range (i, n):
            Z2 = array[0:m, q] 
            #print Z1, Z2
            Z1TZ1 = Z2TZ2 = Z1TZ2 = 0
            for v in range (0, m):
                Z1TZ1 = Z1TZ1 + Z1[v]*Z1[v]
                Z2TZ2 = Z2TZ2 + Z2[v]*Z2[v]
                Z1TZ2 = Z1TZ2 + Z1[v]*Z2[v]
                correlation =  Z1TZ2/(sqrt(Z1TZ1)*sqrt(Z2TZ2))
                if i==q:
                    correlation = 1
            #print degrees(acos(correlation))
            corr_array.append([i,q, correlation, degrees(acos(correlation))])
            print ('The correlation for col: (%d, %d) is %f, and the degree is %.2f' %(i+1, q+1, correlation, degrees(acos(correlation))))
    return corr_array

def least_most(corr_array, array):  
    (m, n) = array.shape
    most_correlated = 180
    most_correlated_set = []
    anti_correlated = 0
    anti_correlated_set = []
    least_correlated = 90
    least_correlated_set = []
    for i in range (0, len(corr_array)):
        if corr_array[i][0]!=corr_array[i][1]:
            if corr_array[i][3]<most_correlated:
                most_correlated = corr_array[i][3]
                most_correlated_set = corr_array[i]
            if corr_array[i][3]>anti_correlated:
                anti_correlated = corr_array[i][3]
                anti_correlated_set = corr_array[i]
            if  abs(corr_array[i][3]-90) <least_correlated:
                least_correlated = abs(corr_array[i][3]-90)
                least_correlated_set =  corr_array[i]
    print ('The most correlated is col: (%d, %d) is %f, and the degree is %.2f' %(most_correlated_set[0]+1, most_correlated_set[1]+1, most_correlated_set[2], most_correlated_set[3]))
    print ('The most anti-correlated is col: (%d, %d) is %f, and the degree is %.2f' %(anti_correlated_set[0]+1, anti_correlated_set[1]+1, anti_correlated_set[2], anti_correlated_set[3]))
    print ('The least correlated is col: (%d, %d) is %f, and the degree is %.2f' %(least_correlated_set[0]+1, least_correlated_set[1]+1, least_correlated_set[2], least_correlated_set[3]))
    plt.figure(1)   
    plt.plot(array[0:m, most_correlated_set[0]], array[0:m, most_correlated_set[1]], 'ro')
    plt.xlabel(most_correlated_set[0]+1)
    plt.ylabel(most_correlated_set[1]+1)
    plt.title('most correlated')   
    plt.savefig('most_correlated.png')
    
    plt.figure(2)  
    plt.plot(array[0:m, anti_correlated_set[0]], array[0:m, anti_correlated_set[1]], 'ro')
    plt.xlabel(anti_correlated_set[0]+1)
    plt.ylabel(anti_correlated_set[1]+1)
    plt.title('most anti-correlated')  
    plt.savefig('anti_correlated.png')    
    
    plt.figure(3)  
    plt.plot(array[0:m, least_correlated_set[0]], array[0:m, least_correlated_set[1]], 'ro')
    plt.xlabel(least_correlated_set[0]+1)
    plt.ylabel(least_correlated_set[1]+1)
    plt.title('least correlated') 
    plt.savefig('least_correlated.png')    
    #print most_correlated_set, anti_correlated_set, least_correlated_set

def eigenvectors(array, e):
   # print array
    (m, n) = array.shape
    X_0 = np.random.random((m,2))
    #X_1 = multiple(X_0, array)
    X_1 =np.dot(array, X_0)
    a = X_1[0:m, 0]
    b = X_1[0:m, 1]
    c =np.dot(b.T, a)/np.dot(a.T, a)
    b = b- c*a
    #print X_1
    X_1[0:m, 1] = b
    error = LA.norm(X_1, 2)
    X_o = X_1
    X_o = X_o/ X_o.sum(axis=0)    
    #print X_o
    while (error>e):
        X_n = np.dot(array, X_o)
        X_n =  X_n/ X_n.sum(axis=0)
        a = X_n[0:m, 0]
        b = X_n[0:m, 1]
        c =np.dot(b.T, a)/np.dot(a.T, a)
        b = b- c*a
        X_n[0:m, 1] = b        
        error = LA.norm(X_n-X_o, 2)
        X_o = X_n
    return X_n    

def plotfigure(array, U):
    (m, n) = array.shape
    (m1, n1) = U.shape
    u1 = U[0:m1, 0]
    u2 = U[0:m1, 1]
    ans = np.zeros((m,n1))
    for i in range (0,m):
        x = array[i, 0:n]
        for v in range (0,n1):
            if v == 0:
                ans[i,v] = np.dot(x.T, u1)
            if v == 1:
                ans[i,v] = np.dot(x.T, u2)   
                
    plt.figure(4)
    plt.plot(ans[0:m, 0], ans[0:m, 1],'bx')
    plt.title('projected points')
    plt.savefig('eigenvectors.png')     
    

        
if __name__ == '__main__':
    print "This is the name of the script: ", sys.argv[0]
    print "Number of arguments: ", len(sys.argv)
    print "The arguments are: " , str(sys.argv) 
    filename = 'airfoil_self_noise.dat'
    filename='Concrete_Data.txt'
    n = 8
    p = 0.0001
    if len(sys.argv) == 4:
        filename = sys.argv[1]
        n = int(sys.argv[2])
        p = float(sys.argv[3])
    if len(sys.argv) == 5:
        orig_stdout = sys.stdout
        f = open('sys.argv[4]', 'w')
        sys.stdout = f            

    np.set_printoptions(precision=n)
    np.set_printoptions(suppress=True)  
    array = readfile(filename)
    #array= np.array([[1, 0.8],[5, 2.4], [9, 5.5] ],dtype=float)
    #print array
    mean_array = mean(array)
    print('The Mean for the data matrix:')
    print mean_array
    total_var = variance(array)
    print('\nThe total variance var(D):')
    print total_var
    inner = inner(array)
    print('\nThe inner products:')
    print inner
    outter = outter(array)
    print('\nThe outter products:')
    print outter
    print('\nThe correlation:')
    corr_array = Correlation(array)
    print('\nThe three correlations:')
    least_most(corr_array, array)
    x = eigenvectors(inner, p)
    print('\nThe eigenvectors:')
    print x
    plotfigure(array, x)
   