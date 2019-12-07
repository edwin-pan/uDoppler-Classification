
import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt
import scipy as sc

plt.close("all")

# Principle Component Analysis Function
def PCA(x, nFeatures):
    """ Perform Principle Component Analysis on an input signal, retaining nFeatures number of features
        This function uses the SVD method of generating PCA, in order to improve computation speed. 
        
        Note that any input matrix X can be written as X = US(V^T)
            U = Eigenvectors of X(X^T) - square matrix (dimensions,dimensions)
            S = Diagonal Matrix of singular values sqrt(\lambda_{i}) - matrix (samples, dimensions) 
                w/ possible  rows or cols of 0's
            V = Eigenvectors of (X^T)X - square matrix (samples, samples)
        
        Eigenvector's of the covariance matrix of mag_spectrum can also be achieved via the Gram matrix.
    
    Args:
        x (np.ndarray): A 2D-array containing the mean centered input signal with shape (dimensions,samples)
        nFeatures (int): The number of features PCA should retain
        
    Returns:
        features (np.ndarray): A 2D array of size (nFeatures, dimensions) containing the feature vectors
        transform (np.ndarray): A 2D array of size (nFeatures, samples) containing the weight vectors
    """

    # Use SVD to grab top nFeatures number of eigenvectors
    u, s, vh = sc.linalg.svd(x)
    # PCA Feature Transform Matrix (Weight matrix)
    transform = vh[:nFeatures,:]
    # Numpy enforces sign conventions for Eigenvectors. Flip negative vectors
    transform[transform[:,0]<0] = -1*transform[transform[:,0]<0]

    # PCA Principle Component (Feature) Vectors np.matmul((3,349),(513,349).T) = np.matmul((3,349),(349,513)) = (3,513)
    features = np.dot(transform, x.T)
    
    return features, transform
