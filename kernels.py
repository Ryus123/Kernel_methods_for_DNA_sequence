#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the computation
of different kernel functions

Created on 12/02/24

Last update 12/02/24 (E.)

@author: E. DELAR
"""

### --- Import
import numpy as np

### --- Class

class Kernels:
    
    def __init__(self):
        self.gram_matrix = None # Rmk : May be store only the triangular version
        self.kernel = None
        
    def Linear(self, x:np.array, y:np.array):
        self.gram_matrix = np.dot(x, y.T)
        self.kernel = 'Linear'
    
    def Min(self, x:np.array, y:np.array):
        self.gram_matrix = np.minimum(x, y)
        self.kernel = 'Min'
        
    def pairwise_distance(self, i, j):
        if i > j :
            return self.gram_matrix[i,j]
        elif i <= j:
            return self.gram_matrix[j, i]
        
    def save_gram_matrix(self):
        np.save('save_Gram/'+self.kernel+'.npy', self.gram_matrix)
        
        
    
        
        

