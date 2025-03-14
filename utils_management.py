#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains utils management 
(load, clean, encoding, plot,..) function.

Created on 12/02/24

Last update 12/02/24 (E.)

@author: E. DELAR
"""

### --- Import
import pandas as pd
import numpy as np

### --- Function
def load_set(path:str):
    """Load a sequence

    Args:
        path (str): sequence file name

    Returns:
        pd.dataframe : dataframe with columns Id, seq
    """
    mat_100 = (len(path.split("_")) == 2) #If it is the mat100 set
    
    if mat_100:
        
        return np.loadtxt('data/'+path, delimiter=' ') # Return a np.array and the associade id of each row
    
    else:
        return pd.read_csv('data/'+path) # Return the dataframe