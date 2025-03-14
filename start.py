#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script reproduce and 
export our submission as 
Yte.csv

Created on 12/02/24

Last update 14/03/24 (E.)

@author: E. DELAR
"""

#############################################################
#### Import
#############################################################
from utils_management import load_set
import numpy as np
from kernels import Spectrum
from compute_classifiers import compute_kernel_LR

#############################################################
#### Load data
#############################################################
print('Load training data...')
X0 = load_set('Xtr0.csv')
X1 = load_set('Xtr1.csv')
X2 = load_set('Xtr2.csv')

X0 = np.array(X0["seq"])
X1 = np.array(X1["seq"])
X2 = np.array(X2["seq"])

y0 = load_set('Ytr0.csv')
label0 = np.array(y0['Bound'])

y1 = load_set('Ytr1.csv')
label1 = np.array(y1['Bound'])

y2 = load_set('Ytr2.csv')
label2 = np.array(y2['Bound'])

train_X = np.concatenate((X0, X1, X2), axis=0)
train_label = np.concatenate((label0, label1, label2))
train_label[train_label==0] = -1
label1[label1==0]= -1

print('Load test data...')
Xt0 = load_set('Xte0.csv')
Xt1 = load_set('Xte1.csv')
Xt2 = load_set('Xte2.csv')

Xt0 = np.array(Xt0["seq"])
Xt1 = np.array(Xt1["seq"])
Xt2 = np.array(Xt2["seq"])

X_test = np.concatenate((Xt0, Xt1, Xt2), axis=0)

#############################################################
#### Predict
#############################################################

compute_kernel_LR(train_X, train_label, X_test, KERNEL=Spectrum(5), lbda=.4, export_path="Yte.csv")