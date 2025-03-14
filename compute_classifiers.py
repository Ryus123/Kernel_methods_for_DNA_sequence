#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains allow to
build the classifier and do prediction
for the submission

Created on 12/02/24

Last update 09/03/24 (E.)

@author: E. DELAR
"""

#############################################################
#### Import
#############################################################
import numpy as np
import pandas as pd
import time

from kernels import Linear, RBF, Polynomial, Min, Spectrum

from kernel_SVM import KernelSVC
from kernel_boosting import KernelModelBoosted
from kernel_logistic_regression import Kernel_LogReg

from utils_management import load_set

#############################################################
#### Load data
#############################################################
# print('Load training data')
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

# print('process data')
train_X = np.concatenate((X0, X1, X2), axis=0)
train_label = np.concatenate((label0, label1, label2))
train_label[train_label==0] = -1
label1[label1==0]= -1

print('\nLoad test data')
Xt0 = load_set('Xte0.csv')
Xt1 = load_set('Xte1.csv')
Xt2 = load_set('Xte2.csv')

Xt0 = np.array(Xt0["seq"])
Xt1 = np.array(Xt1["seq"])
Xt2 = np.array(Xt2["seq"])

print('process test data')
X_test = np.concatenate((Xt0, Xt1, Xt2), axis=0)

#############################################################
#### Define function
#############################################################
def compute_kernel_SVC(train_X, train_label, X_test, KERNEL, C=1.):
    ## Fit the model
    t_start = time.time()

    kernel = KERNEL.kernel
    print('Compute Kernel SVC\n')
    svc_custom = KernelSVC(C=C, kernel=kernel, epsilon=1e-8)
    svc_custom.fit(train_X, train_label)

    building_time = time.time() - t_start

    y_fit = svc_custom.predict(train_X)
    print( f'Times : {building_time/60:.2f}min | Train accuracy : {(y_fit == train_label).mean():.5f}\n\n')

    # Compute the submission file
    y_pred = svc_custom.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv("submission.csv", index=False)

    print("Submission exported!")
    

def compute_kernel_LR(train_X, train_label, X_test, KERNEL, lbda=1):
    ## Fit the model
    t_start = time.time()

    kernel = KERNEL.kernel
    # print('Compute Kernel logistic regression\n')
    klr_custom = Kernel_LogReg(kernel=kernel, lbda=lbda)
    klr_custom.fit(train_X, train_label)

    # building_time = time.time() - t_start

    # # y_fit = klr_custom.predict(train_X)
    # # print(np.unique(y_fit, return_counts=True))
    # # print( f'Times : {building_time/60:.2f}min | Train accuracy : {(y_fit == train_label).mean():.5f}\n\n')

    # Compute the submission file
    y_pred = klr_custom.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv("submission.csv", index=False)

    print("Submission exported!")
    
def compute_kernel_Boosting(train_X, train_label, X_test, model, lr = .002):
    ## Fit the model
    t_start = time.time()

    print('Compute Kernel logistic regression\n')
    boosting_model = KernelModelBoosted(model, lr=lr)
    boosting_model.fit(train_X, train_label)

    building_time = time.time() - t_start

    y_fit = boosting_model.predict(train_X)
    print(np.unique(y_fit, return_counts=True))
    print( f'Times : {building_time/60:.2f}min | Train accuracy : {(y_fit == train_label).mean():.5f}\n\n')

    # Compute the submission file
    y_pred = boosting_model.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv("submission.csv", index=False)

    print("Submission exported!")

#############################################################
#### Compute models
#############################################################

#### ----------- With data mat100

# SVC with rbf kernel with sigma = 1.7 and C = 1
# compute_kernel_SVC(train_X, train_label, X_test, KERNEL=RBF(1.7))

# KLR
# compute_kernel_LR(train_X, train_label, X1, KERNEL=RBF(1.7), lbda=.3)

#### ----------- With data sequence

# SVC with Spectrum kernel with C = 1
# compute_kernel_SVC(train_X, train_label, X_test, KERNEL=Spectrum(2))

# compute_kernel_Boosting(train_X, train_label, X_test, KERNEL=Spectrum(5), lbda=.4)

# model = Kernel_LogReg(kernel=Spectrum(5).kernel, lbda=.4)
# compute_kernel_Boosting(train_X, train_label, X_test, model)