# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:26:06 2019

@author: Edwin

http://users.metu.edu.tr/ccandan//pub_dir/Padar_Ertan_Candan_Micro_Doppler_Classification__IEEE_Radar_2016.pdf

Two approaches to classification problem:
    1) Image Classification Problem
    2) Frequency Varying Time Sequence Problem (Spectrogram)
    
"""

import numpy as np
import func.microdoppler_visualizer as mv
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
data = []
trainDataPed = []
trainDataBic = []
trainLabelPed = []
trainLabelBic = []

sets = 5
setSize = 1000
if load_data_flag:
    print("[NOTE] ---------- Loading Data ----------")

    print("[NOTE] Loading Labels")
    testLabelNoCar = sio.loadmat('data/mathworks/test/testLabelNoCar.mat')
    testLabelNoCar = testLabelNoCar[list(testLabelNoCar.keys())[-1]].squeeze()
    print("[NOTE] Label Loading Complete")
    
    # Grab only Single Pedestrian data
    indicesPed = np.where(testLabelNoCar == np.str_('ped    '))[0]     # Get all indices

    # Grab only Single Bike data
    indicesBic = np.where(testLabelNoCar == np.str_('bic    '))[0]     # Get all indices

    print("[NOTE] Loading Time & Frequency Data")
    TF = sio.loadmat('data/mathworks/TF.mat')
    T = TF[list(TF.keys())[-1]].squeeze()
    F = TF[list(TF.keys())[-2]].squeeze()
    print("[NOTE] T F Loading Complete")

    for i in range(sets):
        testDataNoCar = sio.loadmat('data/mathworks/test/testDataNoCar_'+str(i+1)+'.mat')
        data = np.array(testDataNoCar[list(testDataNoCar.keys())[-1]].squeeze()).transpose(2,0,1)
        print("[NOTE] Loaded data subset ", i)
        trainLabelPed.extend(testLabelNoCar[indicesPed[(indicesPed >= (i*setSize)) & (indicesPed < (i+1)*setSize)]])
        trainDataPed.extend(data[indicesPed[(indicesPed >= (i*setSize)) & (indicesPed < (i+1)*setSize)]-(i*setSize)])
    
    trainDataPed = np.array(trainDataPed)
    trainLabelPed = np.array(trainLabelPed)

    print("[NOTE] ---------- Data Loading Complete ----------")

# Check to see the data
#mv.classification_data_visualizer(data,label=testLabelNoCar)

# Check to see the data
mv.classification_data_visualizer(trainDataPed, trainLabelPed)

# =============================================================================
# Image Classification Problem
# =============================================================================
# 1) Gaussian Mixture Model
    # PCA Feature Extraction

# 2) Convolutional Neural Net

# =============================================================================
# Frequency Varying Time Sequence Problem
# =============================================================================
# Hidden Markov Model -- Lecture 13, Slide 39

# 