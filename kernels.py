#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the computation
of differents kernel functions

Created on 12/02/24

Last update 09/03/24 (E.)

@author: E. DELAR
"""

#############################################################
#### Import
#############################################################
import numpy as np

#############################################################
#### Class
#############################################################


### - Classical kernels
class RBF:    
    def __init__(self, sigma):
        self.sigma = sigma
        self.gram_matrix = None
        
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        
        dist_2 = X_norm + Y_norm - 2 * np.dot(X, Y.T) 
        dist_2 /= 2 * (self.sigma ** 2) + 1e-10
        self.gram_matrix = np.exp(- dist_2)
        
        return np.copy(self.gram_matrix)   ## Matrix of shape NxM
        
    def pairwise_distance(self, i, j):
        if i > j :
            return self.gram_matrix[i,j]
        elif i <= j:
            return self.gram_matrix[j, i]
        
    def save_gram_matrix(self):
        np.save('save_Gram/RBF_'+str(self.sigma)+'.npy', self.gram_matrix) 
        

class Polynomial:    
    def __init__(self, degree, gamma=1, c=0):
        self.deg = int(degree)
        self.gamma = gamma
        self.c = c
        self.gram_matrix = None
        
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        formula = self.gamma*(X @ Y.T) + self.c
        self.gram_matrix = np.power(formula, self.deg)

        return np.copy(self.gram_matrix)   ## Matrix of shape NxM
        
    def pairwise_distance(self, i, j):
        if i > j :
            return self.gram_matrix[i,j]
        elif i <= j:
            return self.gram_matrix[j, i]
        
    def save_gram_matrix(self):
        np.save('save_Gram/Poly_'+str(self.deg)+'.npy', self.gram_matrix) 
        
             
class Linear:    
    def __init__(self):
        self.gram_matrix = None
        
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        self.gram_matrix = X @ Y.T 
        return np.copy(self.gram_matrix)   ## Matrix of shape NxM
        
    def pairwise_distance(self, i, j):
        if i > j :
            return self.gram_matrix[i,j]
        elif i <= j:
            return self.gram_matrix[j, i]
        
    def save_gram_matrix(self):
        np.save('save_Gram/Linear.npy', self.gram_matrix) 


class Min:    
    def __init__(self):
        self.gram_matrix = None
        
    def kernel(self, x:np.array, y:np.array):
        self.gram_matrix = np.minimum(x, y)
        
        return np.copy(self.gram_matrix) ## Matrix of shape NxM
        
    def pairwise_distance(self, i, j):
        if i > j :
            return self.gram_matrix[i,j]
        elif i <= j:
            return self.gram_matrix[j, i]
        
    def save_gram_matrix(self):
        np.save('save_Gram/Min.npy', self.gram_matrix)
    

### - Kernels for biological sequences
        
class Spectrum:    
    def __init__(self, k):
        self.k = k
        self.gram_matrix = None
            
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N, M = X.shape[0], Y.shape[0]
        K = np.zeros((N, M))
        print('\nCompute Spectrum kernel...(Time consuming)')
        for i in range(N):
            substr_x, phi_x = np.unique([X[i][l:l+self.k] for l in range(len(X[i])-self.k+1)], return_counts=True)

            if N != M:
                for j in range(M):
                    K[i, j] = np.sum(np.array( np.char.count(Y[j], substr_x))*phi_x)
            else:
                for j in range(i, M):  
                    K[i, j] = np.sum(np.array(np.char.count(Y[j], substr_x))*phi_x)
                    if i != j:
                        K[j, i] = K[i, j]
            
            # Show progress        
            if i%1000 == 0:
                print(f'{i/N:.2f} % done')
        
        return K   ## Matrix of shape NxM
        
    def pairwise_distance(self, i, j):
        if i > j :
            return self.gram_matrix[i,j]
        elif i <= j:
            return self.gram_matrix[j, i]
        
    def save_gram_matrix(self, path):
        np.save(path, self.gram_matrix) 