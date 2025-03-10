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

class Kernel_LogReg:
    
    def __init__(self, kernel, lbda = 1):
        self.lbda = lbda
        self.kernel = kernel
        self.gram_matrix = None        
        self.alpha = None
        self.train = None
        
    def fit(self, X, y, max_iter = 100, threshold = 1e-6):
        N = X.shape[0]
        self.gram_matrix = self.kernel(X,X)
        self.gram_matrix -= np.mean(self.gram_matrix, axis=0)
        self.gram_matrix /= np.std(self.gram_matrix, axis=0)
        self.alpha = np.ones(N)
        self.train = X
        
        def loss(alpha):
            # Compute the logistique loss
            return np.log(1 + np.exp(-y*(self.gram_matrix @ alpha)))
        
        def grad_loss(alpha):
            # Compute the gradient of the logistique loss
            u = y*(self.gram_matrix @ alpha)
            return (1/(1 + np.exp( -np.clip(u, -200, 200) ) )) - 1
        
        def hess_loss(alpha):
            # Compute the Hessian of the logistique loss
            u = y*(self.gram_matrix @ alpha)
            sigma_u = (1/(1+np.exp(-np.clip(u, -200, 200))))
            return sigma_u*(1 - sigma_u)
        
        # Solving KLR by IRLS
        for _ in range(max_iter):
            m = self.gram_matrix @ self.alpha
            ym = y*m
            P = grad_loss(ym)
            W = hess_loss(ym)
            z = m - (P*y/(W + 1e-10))
            
            # WKRR solution for (SKS +lambda*n*I_n)a = Sy
            A = (self.gram_matrix * W) + (self.lbda*np.eye(N))
            alpha = np.linalg.inv(A) @ self.gram_matrix*W @ z
            
            if ((alpha-self.alpha)**2 <= threshold).all():
                self.alpha = alpha
                print(f'have converged at {_} step')
                break
            
            self.alpha = alpha
    
    def predict(self, X, threshold=0):
        """
        X: array (n_samples, n_features)\\
        Return: float array (n_samples,)
        """
        K = self.kernel(X, self.train)
        K -= np.mean(K, axis=0)
        K /= np.std(K, axis=0)
        y = np.dot(K, self.alpha)
        
        return np.where(y >= 0, 1, -1)