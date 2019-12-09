import numpy as np
from scipy.ndimage import convolve


def multivariate_normal_distrubution(x, mean, covariance):
    """ Given inputs, mean, and covariance, return a 2D-Gaussian distribution
    
    Args:
        x (np.array): a 2D-numpy array containing input positions for plotting
        mean (np.array): a 1D-numpy array containing means
        covariance (np.array): a 2D-array containing covariances
        
    Returns:
        distrubution (np.array): The resulting multivariate normal distribution
    """
    
    inverse = np.linalg.inv(covariance)
    determinant = np.linalg.det(covariance)
    scale = 1 / (np.sqrt((2*np.pi)**mean.shape[0]*determinant))
    exponent = np.exp(-1*np.einsum('...k,kl,...l->...', x-mean, inverse, x-mean)/2)        
    
    return scale*exponent


def downsampler_2(x):
    """ Downsample a given 3D nd array of images with shape (nimages, orig_x, orig_y) by a factor of 2
    
    Args:
        x (np.array): A 3D-numpy array containing signal

    Returns:
        output (np.array): nimages number of downsampled spectrograms
    """
    
    num_img, orig_x, orig_y = x.shape
    output = np.zeros((num_img,orig_x//2, orig_y//2),dtype=np.float32)
    for i in range(num_img):
        output[i] = convolve(x[i], np.array([[0.25,0.25],[0.25,0.25]]))[:orig_x:2,:orig_y:2]
    return output