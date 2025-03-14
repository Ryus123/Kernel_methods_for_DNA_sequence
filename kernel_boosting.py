#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script implement the 
approach from the paper
'Boosting as a Kernel-Based
Method' from Aravkin et al.

    - https://arxiv.org/pdf/1608.02485

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

class KernelModelBoosted:
    
    def __init__(self, model, lr = .002):
        self.model = model
        self.lr = lr
        self.learned_model = []
        self.init = None
    
    def calcul_proba(self, score):
        score = np.clip(score, -200, 200)
        return 1/(1+np.exp(-score))
    
    def fit(self, X, y, max_iter = 3, threshold = 1e-6):
        N = X.shape[0]
        y = (y+1)/2
        self.init = np.log(np.mean(y) / (1 - np.mean(y)))
        y_fit = np.ones(N)*self.init
        
        model_boost = self.model
        model_boost.fit(X,y)
        self.learned_model.append(model_boost)
        # Fit model on the residuals
        for i in range(max_iter):
            print(f'Train boosting step {i+1}')
            score = self.model.score(X)
            proba = self.calcul_proba(score)
            res = - (y - proba)
            
            model_boost = self.model
            model_boost.fit(X,res)
            y_fit -= self.lr * model_boost.score(X)
            
            self.learned_model.append(model_boost)
            
    
    def predict(self, X, threshold=.5):
        n_pred = X.shape[0]
        score_pred = np.ones(n_pred)*self.init
        
        for mdl in self.learned_model:
            score_pred -= self.lr * mdl.score(X)
        
        proba_pred = self.calcul_proba(score_pred)
        
        return np.where(proba_pred >= threshold, 1, -1)
        