# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:26:06 2019

@author: Edwin
"""

import numpy as np
import func.capon as lc
import mmwave.dsp as dsp
import scipy.io as sio
    
    
#testDataCarNoise = sio.loadmat('data/mathworks/testDataCarNoise')
#trainDataCarNoise = sio.loadmat('data/mathworks/trainDataCarNoise')
#trainDataNoCar = sio.loadmat('data/mathworks/trainDataCarNoise')

# =============================================================================
# Load data
# =============================================================================
print("[NOTE] Loading Data")
testDataNoCar_1 = sio.loadmat('data/mathworks/test/testDataNoCar_1.mat')
testDataNoCar_1 = testDataNoCar_1[list(testDataNoCar_1.keys())[-1]]
print("[NOTE] Loaded data subset 1")

testDataNoCar_2 = sio.loadmat('data/mathworks/test/testDataNoCar_2.mat')
testDataNoCar_2 = testDataNoCar_2[list(testDataNoCar_2.keys())[-1]]
print("[NOTE] Loaded data subset 2")

testDataNoCar_3 = sio.loadmat('data/mathworks/test/testDataNoCar_3.mat')
testDataNoCar_3 = testDataNoCar_3[list(testDataNoCar_3.keys())[-1]]
print("[NOTE] Loaded data subset 3")

testDataNoCar_4 = sio.loadmat('data/mathworks/test/testDataNoCar_4.mat')
testDataNoCar_4 = testDataNoCar_4[list(testDataNoCar_4.keys())[-1]]
print("[NOTE] Loaded data subset 4")

testDataNoCar_5 = sio.loadmat('data/mathworks/test/testDataNoCar_5.mat')
testDataNoCar_5 = testDataNoCar_5[list(testDataNoCar_5.keys())[-1]]
print("[NOTE] Loaded data subset 5")

print("[NOTE] Data Loading Complete")

print("[NOTE] Loading Labels")
testLabelNoCar = sio.loadmat('data/mathworks/test/testLabelNoCar.mat')
testLabelNoCar = testLabelNoCar[list(testLabelNoCar.keys())[-1]] # Bug
print("[NOTE] Label Loading Complete")
