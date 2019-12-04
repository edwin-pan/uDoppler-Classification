
import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt

plt.close("all")

def ICA(Z_PCA, nFeatures, rate, threshold, identity=True):
    """ Achieves Independent Component Analysis via leveraging the results from PCA, and calculating a 
        mixing matrix (W_ICA) which, when applied the PCA's output feature vector, will transform the
        PCA's decorrelated feature vectors into independent feature vectors.
        
    Args:
        Z_PCA (np.ndarray): A 2D-array containing the output weight vectors from PCA
        rate (float): A constant to define the learning rate when iteratively learning optimal W_ICA
        
    Returns:
        W_ICA (np.ndarray): A 2D-array (square matrix) containing transform coefficients that will convert
                            the PCA feature vector into ICA feature vector via 
    
    """
    # Starting ICA mixing matrix can either be identity matrix or random diagonal matrix
    if identity:
        W_ICA = np.identity(nFeatures) 
    else:
        W_ICA = np.diag(np.random.rand(3))

    # Calculate initial y
    y = np.matmul(W_ICA, Z_PCA)

    iterationCount = 0
    deltaW = np.inf

    # Iteratively compute new mixing matrices until change per iteration decreases below threshold
    print(Z_PCA.shape[1])
    while np.linalg.norm(deltaW) > threshold:
        # Calculate new deltaW
        deltaW = rate * np.matmul(Z_PCA.shape[1]*np.identity(nFeatures)-np.matmul(2*np.tanh(y),y.T),W_ICA)
        # Adjust mixing matrix with new deltaW
        W_ICA += deltaW
        # Calculate new y
        y = np.matmul(W_ICA, Z_PCA)
        
        iterationCount+=1

    print("[ICA] ICA mixing matrix found in ", iterationCount, " iterations.")
    return W_ICA