# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:26:06 2019

@author: Edwin
"""

import numpy as np
import func.microdoppler_visualizer as mv
import mmwave.dsp as dsp
import scipy.io as sio
import matplotlib.pyplot as plt
    
#testDataCarNoise = sio.loadmat('data/mathworks/testDataCarNoise')
#trainDataCarNoise = sio.loadmat('data/mathworks/trainDataCarNoise')
#trainDataNoCar = sio.loadmat('data/mathworks/trainDataCarNoise')
plt.close('all')
load_data_flag = True
# =============================================================================
# Load data
# =============================================================================
if load_data_flag:
    print("[NOTE] Loading Data")
    testDataNoCar_1 = sio.loadmat('data/mathworks/test/testDataNoCar_1.mat')
    testDataNoCar_1 = testDataNoCar_1[list(testDataNoCar_1.keys())[-1]].squeeze()
    print("[NOTE] Loaded data subset 1")
    
#    testDataNoCar_2 = sio.loadmat('data/mathworks/test/testDataNoCar_2.mat')
#    testDataNoCar_2 = testDataNoCar_2[list(testDataNoCar_2.keys())[-1]].squeeze()
#    print("[NOTE] Loaded data subset 2")
#    
#    testDataNoCar_3 = sio.loadmat('data/mathworks/test/testDataNoCar_3.mat')
#    testDataNoCar_3 = testDataNoCar_3[list(testDataNoCar_3.keys())[-1]].squeeze()
#    print("[NOTE] Loaded data subset 3")
#    
#    testDataNoCar_4 = sio.loadmat('data/mathworks/test/testDataNoCar_4.mat')
#    testDataNoCar_4 = testDataNoCar_4[list(testDataNoCar_4.keys())[-1]].squeeze()
#    print("[NOTE] Loaded data subset 4")
#    
#    testDataNoCar_5 = sio.loadmat('data/mathworks/test/testDataNoCar_5.mat')
#    testDataNoCar_5 = testDataNoCar_5[list(testDataNoCar_5.keys())[-1]].squeeze()
#    print("[NOTE] Loaded data subset 5")
    
    print("[NOTE] Data Loading Complete")
    
    print("[NOTE] Loading Labels")
    testLabelNoCar = sio.loadmat('data/mathworks/test/testLabelNoCar.mat')
    testLabelNoCar = testLabelNoCar[list(testLabelNoCar.keys())[-1]].squeeze()
    print("[NOTE] Label Loading Complete")
    
    print("[NOTE] Loading Time & Frequency Data")
    TF = sio.loadmat('data/mathworks/TF.mat')
    T = TF[list(TF.keys())[-1]].squeeze()
    F = TF[list(TF.keys())[-2]].squeeze()
    print("[NOTE] T F Loading Complete")

# Check to see the data
mv.classification_data_visualizer(testDataNoCar_1.transpose(2,0,1), testLabelNoCar)
# =============================================================================
# 
# =============================================================================
    
    
#fig, ax = plt.subplots(figsize=(7,7))
#img = ax.imshow(testDataNoCar_1[:,:,21])
#fig.colorbar(img)
#ax.set_xticks(list(range(1,T.shape[0],36)))
#ax.set_yticks(list(range(1,F.shape[0],100)))
#ax.set_xticklabels(T[list(range(1,T.shape[0],36))])
#ax.set_yticklabels(F[list(range(1,F.shape[0],100))])
