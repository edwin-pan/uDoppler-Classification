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
import func.pca as pca
import scipy.io as sio
import matplotlib.pyplot as plt
    
plt.close('all')

# Load data
trainDataPed = np.load('data/mathworks/test/test_data_ped.npy')
trainDataBic = np.load('data/mathworks/test/test_data_bic.npy')
trainLabelPed = np.load('data/mathworks/test/test_label_ped.npy') 
trainLabelBic = np.load('data/mathworks/test/test_label_bic.npy')

# Vectorize "image" data
trainDataPedVec = trainDataPed.reshape((trainDataPed.shape[0],-1), order='F')
trainDataBicVec = trainDataBic.reshape((trainDataBic.shape[0],-1), order='F')

# =============================================================================
# Image Classification Problem
# =============================================================================
# 1) Gaussian Mixture Model
nFeatures = 2
    # PCA Feature Extraction -- Compute Features via PCA using Mean Centered Ped & Bic spectrograms
trainDataPedVecWeights, trainDataPedVecFeatures = pca.PCA(trainDataPedVec-np.mean(trainDataPedVec, axis=0), nFeatures)
trainDataBicVecWeights, trainDataBicVecFeatures = pca.PCA(trainDataBicVec-np.mean(trainDataBicVec, axis=0), nFeatures)


# 2) Convolutional Neural Net


# =============================================================================
# Frequency Varying Time Sequence Problem
# =============================================================================
# Hidden Markov Model -- Lecture 13, Slide 39

# 