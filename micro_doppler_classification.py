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
import torch
import func.microdoppler_visualizer as mv
import func.pca as pca
import func.utils as utils
import models.ConvolutionalNeuralNetwork as CNN
import models.GaussianMixtureModel as GMM
import scipy.io as sio
import matplotlib.pyplot as plt
import argparse

plt.close('all')

nSamplesPerClass = 1000

def main(args):
# Load data
    trainDataPed = np.load('data/processed/test_data_ped.npy').astype(np.float32)
    trainDataBic = np.load('data/processed/test_data_bic.npy').astype(np.float32)
    trainLabelPed = np.load('data/processed/test_label_ped.npy')
    trainLabelBic = np.load('data/processed/test_label_bic.npy')

    testDataPed = np.load('data/processed/train_data_ped.npy').astype(np.float32)
    testDataBic = np.load('data/processed/train_data_bic.npy').astype(np.float32)
    testLabelPed = np.load('data/processed/train_label_ped.npy')
    testLabelBic = np.load('data/processed/train_label_bic.npy')

    # Downsample by a factor of 2
    trainDataPed_ds = utils.downsampler_2(trainDataPed)
    trainDataBic_ds = utils.downsampler_2(trainDataBic)
    testDataPed_ds = utils.downsampler_2(testDataPed)
    testDataBic_ds = utils.downsampler_2(testDataBic)

    # Vectorize "image" data
    trainDataPedVec = trainDataPed_ds.reshape((trainDataPed_ds.shape[0],-1), order='F')
    trainDataBicVec = trainDataBic_ds.reshape((trainDataBic_ds.shape[0],-1), order='F')
    testDataPedVec = testDataPed_ds.reshape((testDataPed_ds.shape[0],-1), order='F')
    testDataBicVec = testDataBic_ds.reshape((testDataBic_ds.shape[0],-1), order='F')

    # Check out the Downsampled data
    #mv.classification_data_visualizer(trainDataPedVec.reshape((trainDataPedVec.shape[0],200,72), order='F'), trainLabelPed)

    ## # --- Use for Feature Plots (PCA)
    #nFeatures = 16
    ## PCA Feature Extraction -- Compute Features via PCA using Mean Centered Ped & Bic spectrograms
    #trainDataPedVecWeights, trainDataPedVecFeatures = pca.PCA(trainDataPedVec-np.mean(trainDataPedVec, axis=0), nFeatures)
    #trainDataBicVecWeights, trainDataBicVecFeatures = pca.PCA(trainDataBicVec-np.mean(trainDataBicVec, axis=0), nFeatures)
    #
    ## NMF Feature Extraction -- Compute Features via NMF using Mean Centered Ped & Bic spectrograms
    ## trainDataPedVecWeightsNMF, trainDataPedVecFeaturesNMF = pca.PCA(trainDataPedVec-np.mean(trainDataPedVec, axis=0), nFeatures)
    ## trainDataBicVecWeightsNMF, trainDataBicVecFeaturesNMF = pca.PCA(trainDataBicVec-np.mean(trainDataBicVec, axis=0), nFeatures)
    #
    ## Check out the features
    ## mv.classification_data_visualizer(trainDataPedVecFeatures.reshape((nFeatures,200,72), order='F'), np.array([str(i) for i in range(nFeatures)]))
    #mv.feature_viewer(trainDataPedVecFeatures.reshape((nFeatures,200,72), order='F'),nFeatures, trainDataPed_ds.shape[1], trainDataPed_ds.shape[2], title='Pedestrian Features')
    #mv.feature_viewer(trainDataBicVecFeatures.reshape((nFeatures,200,72), order='F'),nFeatures, trainDataBic_ds.shape[1], trainDataBic_ds.shape[2], title='Bike Features')
    ## # --- Use for Feature Plots (PCA)


    # =============================================================================
    # 1) Gaussian Mixture Model
    # =============================================================================
    if args.model == 'GMM':
        print("[GMM] Begin GMM Training & Testing")
        nFeatures = 16
        nClasses = 2

        # Produce full set
        fullSet = np.concatenate((trainDataPedVec,trainDataBicVec), axis=0)
        fullSetLabel = np.concatenate((trainLabelPed,trainLabelBic), axis=0)

        trainingDataMean = np.mean(fullSet, axis=0)
        weights, features = pca.PCA(fullSet-trainingDataMean, nFeatures)
        
        if args.see_features:
            print(args.see_features)
            mv.feature_viewer(features.reshape((nFeatures,200,72), order='F'),nFeatures, trainDataBic_ds.shape[1], trainDataBic_ds.shape[2], title='GMM Features')
        if args.see_weights:
            print(args.see_weights)
            mv.weight_viewer(weights, fullSetLabel)

        # Generate mean and covariance for bike and pedestrian class
        gmm_classifier = GMM.GaussianMixtureModel(fullSet, nFeatures, 2, 1000)

        results = gmm_classifier.fit(fullSet)

        # Make a decision
        decision = np.argmax(results, axis=0)
        decisionLabeled = []
        for sample in decision:
            if sample == 0:
                decisionLabeled.append('ped    ')
            elif sample == 1:
                decisionLabeled.append('bic    ')
        decisionLabeled = np.array(decisionLabeled)

        # Calculate Statistics
        train_accuracy = np.mean(decisionLabeled == fullSetLabel)
        print("Training set accuracy: ", train_accuracy)

        # -- Now test
        testFullSet = np.concatenate((testDataPedVec,testDataBicVec), axis=0)
        testFullSetLabel = np.concatenate((testLabelPed,testLabelBic), axis=0)

        # Generate mean and covariance for bike and pedestrian class
        testResults = gmm_classifier.fit(testFullSet)

        # Make a decision
        testDecision = np.argmax(testResults, axis=0)
        testDecisionLabeled = []
        for sample in testDecision:
            if sample == 0:
                testDecisionLabeled.append('ped    ')
            elif sample == 1:
                testDecisionLabeled.append('bic    ')
        testDecisionLabeled = np.array(testDecisionLabeled)

        # Calculate Statistics
        bike_correct = 0
        bike_incorrect = 0
        ped_correct = 0
        ped_incorrect = 0
        for i in range(len(testDecisionLabeled)):
            if testDecisionLabeled[i] == 'ped    ':
                if testDecisionLabeled[i] == testFullSetLabel[i]:
                    ped_correct +=1
                else:
                    ped_incorrect +=1
            else:
                if testDecisionLabeled[i] == testFullSetLabel[i]:
                    bike_correct +=1
                else:
                    bike_incorrect +=1

        test_accuracy = np.mean(testDecisionLabeled == testFullSetLabel)
        print("[GMM] Testing set accuracy: ", test_accuracy)

    # =============================================================================
    # 2) Convolutional Neural Net
    # =============================================================================
    elif args.model == 'CNN':
        print("[CNN] Begin CNN Training & Testing")
        # Produce train set
        trainSet = np.concatenate((trainDataPed_ds,trainDataBic_ds), axis=0)
        trainSetLabel = np.concatenate((trainLabelPed,trainLabelBic), axis=0)
        # Convert train set to torch tensor
        trainSet = torch.tensor(trainSet,dtype=torch.float32)
        # Answer to the question: Is it a bike?
        trainSetLabel_binary = np.array([int('bic    '==elem) for elem in trainSetLabel])
        trainSetLabel_bool = np.array([('bic    '==elem) for elem in trainSetLabel])

        # Produce test set
        testSet = np.concatenate((testDataPed_ds,testDataBic_ds), axis=0)
        testSetLabel = np.concatenate((testLabelPed,testLabelBic), axis=0)
        # Convert test set to torch tensor
        testSet = torch.tensor(testSet,dtype=torch.float32)
        # Answer to the question: Is it a bike?
        testSetLabel_bool = np.array(['bic    '==elem for elem in testSetLabel])

        train_flag = False

        if train_flag:
            _, net = CNN.fit(trainSet, trainSetLabel_binary, testSet, 10)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
            in_size = 0
            out_size = 2
            net = CNN.NeuralNet(0.03, loss_fn, in_size, out_size)
            net.load_state_dict(torch.load('net.model'))

        net.eval()
        batch_size = 10

        # Begin - Train
        num_batch_train = trainSet.shape[0]//batch_size
        result_train = np.zeros((num_batch_train*batch_size,2))

        # Evaluate - Train
        for i in range(num_batch_train):
            result_train[i*10:(i+1)*10] = net(trainSet[i*10:(i+1)*10]).detach().numpy()

        # Decide - Train
        decision_train = np.array([sample[0]<sample[1] for sample in result_train])
        train_accuracy = np.mean(decision_train == trainSetLabel_bool)
        print("[CNN] Training set accuracy: ", train_accuracy)  
            
        # Begin - Test
        num_batch = testSet.shape[0]//batch_size
        result = np.zeros((num_batch*batch_size,2))

        # Evaluate - Test
        for i in range(num_batch):
            result[i*10:(i+1)*10] = net(testSet[i*10:(i+1)*10]).detach().numpy()
            
        # Decide - Test
        decision = np.array([sample[0]<sample[1] for sample in result])
        test_accuracy = np.mean(decision == testSetLabel_bool)
        print("[CNN] Testing set accuracy: ", test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MicroDoppler Classification Main')
    parser.add_argument('model', choices=['GMM', 'CNN'])
    parser.add_argument('--see_features', action='store_true', help='Allows user to view GMM features')
    parser.add_argument('--see_weights', action='store_true', help='Allows user to view GMM weights')

    # Call
    args = parser.parse_args()
    main(args)