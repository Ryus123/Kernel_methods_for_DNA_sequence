#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the computation
of the Kernel SVM method

Created on 12/02/24

Last update 09/03/24 (E.)

@author: E. DELAR
"""

#############################################################
#### Import
#############################################################
import numpy as np
from scipy import optimize
import cvxopt

#############################################################
#### Class
#############################################################

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None # support vectors
        self.epsilon = epsilon
        self.norm_f = None
    
    def dual_solver(self, N, y):
        #Set option
        cvxopt.solvers.options['show_progress'] = False
        # Define the matricies
        Y = np.diag(y)
        P = cvxopt.matrix(Y @ self.Gram_matrix @ Y)
        q = cvxopt.matrix( - np.ones(N, dtype=np.float64))
        G = cvxopt.matrix(np.vstack([- np.eye(N), np.eye(N)]))
        h= cvxopt.matrix(np.hstack([np.zeros(N), self.C*np.ones(N)], dtype=np.float64))
        A = cvxopt.matrix(y.astype(float)).T
        b = cvxopt.matrix([0.])
        # Define the quadratic programme
        sol = cvxopt.solvers.qp(P,q,G,h,A,b)
        # Assigne the solution to alpha
        self.alpha = np.array(sol['x']).T[0]
        
    def fit(self, X, y):
        N = len(y)
        self.Gram_matrix = self.kernel(X,X)
        ### Find the optimal solution with cvxopt
        
        # Dual problem
        self.dual_solver(N, y)

        ## Assign the required attributes
        v = self.alpha*y
        mask = self.alpha > self.epsilon
        self.v_support = v[mask]
        
        self.support = X[mask]  #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        self.b = np.mean( y[mask] - np.sum(self.v_support * (self.Gram_matrix[:, mask]), axis=0) )  #''' -----------------offset of the classifier------------------ '''
        self.norm_f = v.T @ self.Gram_matrix @ v #'''------------------- A matrix with each row corresponding to support vectors ------------------'''
        

    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        sep = self.kernel(x, self.support) @ self.v_support
        return sep
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1