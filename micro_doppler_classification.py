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
import func.utils as utils
import scipy.io as sio
import matplotlib.pyplot as plt
    
plt.close('all')

nSamples = 100

# Load data
trainDataPed = np.load('data/mathworks/test/test_data_ped.npy').astype(np.float32)[:nSamples]
trainDataBic = np.load('data/mathworks/test/test_data_bic.npy').astype(np.float32)[:nSamples]
trainLabelPed = np.load('data/mathworks/test/test_label_ped.npy')[:nSamples]
trainLabelBic = np.load('data/mathworks/test/test_label_bic.npy')[:nSamples]

# Downsample by a factor of 2
trainDataPed_ds = utils.downsampler_2(trainDataPed)
trainDataBic_ds = utils.downsampler_2(trainDataBic)

# Vectorize "image" data
trainDataPedVec = trainDataPed_ds.reshape((trainDataPed_ds.shape[0],-1), order='F')
trainDataBicVec = trainDataBic_ds.reshape((trainDataBic_ds.shape[0],-1), order='F')

# Check out the Downsampled data
# mv.classification_data_visualizer(trainDataPedVec.reshape((nSamples,200,72), order='F'), trainLabelPed)

# =============================================================================
# Image Classification Problem
# =============================================================================
# 1) Gaussian Mixture Model
nFeatures = 16
    # PCA Feature Extraction -- Compute Features via PCA using Mean Centered Ped & Bic spectrograms
trainDataPedVecWeights, trainDataPedVecFeatures = pca.PCA(trainDataPedVec-np.mean(trainDataPedVec, axis=0), nFeatures)
trainDataBicVecWeights, trainDataBicVecFeatures = pca.PCA(trainDataBicVec-np.mean(trainDataBicVec, axis=0), nFeatures)

    # NMF Feature Extraction -- Compute Features via NMF using Mean Centered Ped & Bic spectrograms
# trainDataPedVecWeightsNMF, trainDataPedVecFeaturesNMF = pca.PCA(trainDataPedVec-np.mean(trainDataPedVec, axis=0), nFeatures)
# trainDataBicVecWeightsNMF, trainDataBicVecFeaturesNMF = pca.PCA(trainDataBicVec-np.mean(trainDataBicVec, axis=0), nFeatures)

# Check out the features
# mv.classification_data_visualizer(trainDataPedVecFeatures.reshape((nFeatures,200,72), order='F'), np.array([str(i) for i in range(nFeatures)]))
mv.feature_viewer(trainDataPedVecFeatures.reshape((nFeatures,200,72), order='F'),nFeatures, trainDataPed_ds.shape[1], trainDataPed_ds.shape[2], title='Pedestrian Features')
mv.feature_viewer(trainDataBicVecFeatures.reshape((nFeatures,200,72), order='F'),nFeatures, trainDataBic_ds.shape[1], trainDataBic_ds.shape[2], title='Bike Features')

# 2) Convolutional Neural Net


# =============================================================================
# Frequency Varying Time Sequence Problem
# =============================================================================
# Hidden Markov Model -- Lecture 13, Slide 39

# 