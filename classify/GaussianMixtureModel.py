import numpy as np
import func.pca as pca
import func.utils as utils

class GaussianMixtureModel():
    def __init__(self, trainingData, nFeatures, nClasses, nSamplesPerClass):
        """ Initialize the Gaussian Mixture Model by calculating feature vectors for training data, and generating gaussians

        Args:
            trainingData (np.array): A 2D numpy array of vectorized microdoppler spectrograms with shape (nSamples, nPixels)
            nFeatures (int): Number of features to retain in PCA
            nClasses (int): Number of classes to classify
            nSamplesPerClass (int): Number of samples of each class 

        """
        _, self.nPixels = trainingData.shape
        self.nClasses = nClasses
        self.nSamplesPerClass = nSamplesPerClass
        self.trainingDataMean = np.mean(trainingData, axis=0)
        self.weights, self.features = pca.PCA(trainingData-self.trainingDataMean, nFeatures)

        # Populate means & covariance for all classes
        self.update(trainingData)


    def update(self, trainingData):
        """ Gaussian distributions are uniquely identifiable by their mean and covariance. This function generates nFeature means
        and covariances for the two Multivariate Gaussian Distributions.

        Args:
            trainingData (np.array): A 2D numpy array of vectorized microdoppler spectrograms with shape (nSamples, nPixels)

        """
        self.means = []
        self.covar = []

        for i in range(self.nClasses):
            self.means.append(self.weights[:,i*self.nSamplesPerClass:(i+1)*self.nSamplesPerClass].mean(axis=1))
            self.covar.append(np.matmul((self.weights[:,i*self.nSamplesPerClass:(i+1)*self.nSamplesPerClass].T-self.means[i]).T,(self.weights[:,i*self.nSamplesPerClass:(i+1)*self.nSamplesPerClass].T-self.means[i])) / (self.weights[:,i*self.nSamplesPerClass:(i+1)*self.nSamplesPerClass].shape[1]-1))


    def fit(self, inputData):
        """ Given input data, calculate probabilities of admittance as one of the classes based on assumed Gaussian Distributed system.

        Args:
            inputData (np.array): A 2D numpy array of vectorized microdoppler spectrograms with shape (nSamples, nPixels)
        """

        # Calculate weights for inputData given features from trainingSet
        inputWeights = np.matmul(self.features, (inputData-self.trainingDataMean).T)

        # Use calculated weights to generate a probability of admittance to each class
        inputNumberOfSamples = inputData.shape[0]
        probabilities = np.zeros((self.nClasses,inputNumberOfSamples))
        for i in range(self.nClasses):
            probabilities[i] = utils.multivariate_normal_distrubution(inputWeights.T, self.means[i], self.covar[i])
            
        return probabilities
