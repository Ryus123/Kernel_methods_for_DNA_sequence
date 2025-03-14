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

from kernel_SVM import KernelSVC
from kernel_boosting import KernelModelBoosted
from kernel_logistic_regression import Kernel_LogReg

#############################################################
#### Define function
#############################################################
def compute_kernel_SVC(train_X, train_label, X_test, KERNEL, C=1., export_path="submission.csv"):

    kernel = KERNEL.kernel
    print('Compute Kernel SVC...\n')
    svc_custom = KernelSVC(C=C, kernel=kernel, epsilon=1e-8)
    svc_custom.fit(train_X, train_label)

    # Compute the submission file
    y_pred = svc_custom.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv(export_path, index=False)

    print("Submission exported!")
    

def compute_kernel_LR(train_X, train_label, X_test, KERNEL, lbda=1, export_path="submission.csv"):

    kernel = KERNEL.kernel
    print('Compute Kernel logistic regression...\n')
    klr_custom = Kernel_LogReg(kernel=kernel, lbda=lbda)
    klr_custom.fit(train_X, train_label)

    # Compute the submission file
    y_pred = klr_custom.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv(export_path, index=False)

    print("Submission exported!")
    
    
def compute_kernel_Boosting(train_X, train_label, X_test, model, lr = .002, export_path="submission.csv"):

    print('Compute Boosting Kernel logistic regression...\n')
    boosting_model = KernelModelBoosted(model, lr=lr)
    boosting_model.fit(train_X, train_label)

    # Compute the submission file
    y_pred = boosting_model.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv(export_path, index=False)

    print("Submission exported!")