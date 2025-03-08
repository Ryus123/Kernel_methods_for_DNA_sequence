#############################################################
#### Import
#############################################################
from kernel_SVM import KernelSVC
from kernels import Linear, RBF, Polynomial
from utils_management import load_set
import numpy as np
import pandas as pd

import time

#############################################################
#### Load data
#############################################################
print('Load training data')
X0 = load_set('Xtr0_mat100.csv')
X1 = load_set('Xtr1_mat100.csv')
X2 = load_set('Xtr2_mat100.csv')

y0 = load_set('Ytr0.csv')
label0 = np.array(y0['Bound'])

y1 = load_set('Ytr1.csv')
label1 = np.array(y1['Bound'])

y2 = load_set('Ytr2.csv')
label2 = np.array(y2['Bound'])


print('process data')
train_X = np.vstack((X0, X1, X2))
train_label = np.concatenate((label0, label1, label2))
train_label[train_label==0] = -1


print('\nLoad test data')
Xt0 = load_set('Xte0_mat100.csv')
Xt1 = load_set('Xte1_mat100.csv')
Xt2 = load_set('Xte2_mat100.csv')


print('process test data')
X_test = np.vstack((Xt0, Xt1, Xt2))

#############################################################
#### Define function
#############################################################
def compute_RBF_SVC(train_X, train_label, X_test):
    ## Fit the model
    print('Compute KernelSVC\n')
    C=1.
    t_start = time.time()

    kernel = RBF(1.7).kernel
    svc_custom = KernelSVC(C=C, kernel=kernel, epsilon=1e-8)
    svc_custom.fit(train_X, train_label)

    building_time = time.time() - t_start

    y_fit = svc_custom.predict(train_X)
    print( f'Times : {building_time/60:.2f}min | Train accuracy : {(y_fit == train_label).mean()}\n\n')

    ## Compute the submission file
    y_pred = svc_custom.predict(X_test)
    y_pred[y_pred==-1] = 0

    df = pd.DataFrame({"Id": np.arange(len(y_pred)), "Bound": y_pred})
    df.to_csv("submission.csv", index=False)

    print("Submission exported!")
    

#############################################################
#### Compute models
#############################################################

# SVC with rbf kernel with sigma = 1.7 and C = 1
compute_RBF_SVC(train_X, train_label, X_test)