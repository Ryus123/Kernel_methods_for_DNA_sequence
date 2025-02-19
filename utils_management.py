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

### --- Function
def load_set(path:str):
    """Load a sequence

    Args:
        path (str): sequence file name

    Returns:
        pd.dataframe : dataframe with columns Id, seq
    """
    return pd.read_csv('data/'+path)


def export_prediction(pred:pd.DataFrame):
    #TO DO
    pass
    
