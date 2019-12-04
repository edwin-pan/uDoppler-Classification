# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 15:49:51 2019

@author: Edwin
"""
import numpy as np


def range_selection(rangeDoppler, threshold=0.8):
    """ --- NEEDS TO BE CLEANED & OPTIMIZED
    Finds range of ranges that contain the peak, within threshold. Assumes single peak
    
    Args:
        rangeDoppler (~numpy.ndarray): RDC organized as (numFrame, numRangeBins, numDopplerBins)
        
        threshold(float): threshold of max peak height for range finding
        
    Returns:
        (~numpy.ndarray) with min, mid, max of range of ranges (numFrames,3)
    
    """
    # Begin Range Stitching (single target) Procedure
    peaks = []
    maxes = []
    thresh = []

    ranges = []
    dopplers = []
    for i in range(rangeDoppler.shape[0]):
        peaks.append(np.unravel_index(np.argmax(rangeDoppler[i],axis=None), rangeDoppler[i].shape))
        coord_R = peaks[-1][0]
        coord_D = peaks[-1][1]
        maxes.append(rangeDoppler[i][coord_R][coord_D])
        thresh.append(threshold*maxes[-1])
        
        # Find the RANGE of values that are within threshold
        idx_1 = 0
        while rangeDoppler[i][coord_R-idx_1][coord_D] > thresh[-1]:
            idx_1 += 1
            if coord_R-idx_1 < 0:
                break
        idx_2 = 0
        while rangeDoppler[i][coord_R+idx_2][coord_D] > thresh[-1]:
            idx_2 += 1
            if coord_R+idx_2 >= rangeDoppler.shape[1]:
                break
        ranges.append((coord_R-idx_1, coord_R, coord_R+idx_2))
    
        # Find the DOPPLER  of values that are within threshold
        idx_3 = 0
        while rangeDoppler[i][coord_R][coord_D-idx_3] > thresh[-1]:
            idx_3 += 1
            if coord_R-idx_3 < 0:
                break
        idx_4 = 0
        while rangeDoppler[i][coord_R][coord_D+idx_4] > thresh[-1]:
            idx_4 += 1
            if coord_R+idx_4 >= rangeDoppler.shape[1]:
                break
        dopplers.append((coord_D-idx_3, coord_D, coord_D+idx_4))
        
    rangeRanges = np.array(ranges)
    uDoppler_stitched = -7*np.ones((rangeDoppler.shape[2],rangeDoppler.shape[0]))

    for i in range(rangeDoppler.shape[0]):
        lower = rangeRanges[i,0]
        upper = rangeRanges[i,2]
        uDoppler_stitched[:,i] = np.sum(rangeDoppler[i,lower:upper,:], axis=0)/(upper-lower)
    
    return uDoppler_stitched


def range_doppler_selection(rangeDoppler, threshold=0.8):
    """ --- NEEDS TO BE CLEANED & OPTIMIZED
    Finds range of ranges that contain the peak, within threshold. Assumes single peak
    
    Args:
        rangeDoppler (~numpy.ndarray): RDC organized as (numFrame, numRangeBins, numDopplerBins)
        
        threshold(float): threshold of max peak height for range finding
        
    Returns:
        (~numpy.ndarray) with min, mid, max of range of ranges (numFrames,3)
    
    """
    # Begin Range Stitching (single target) Procedure

    ranges = []
    for i in range(rangeDoppler.shape[0]):
        peak_coord = np.unravel_index(np.argmax(rangeDoppler[i],axis=None), rangeDoppler[i].shape)
        coord_R = peak_coord[0]
        coord_D = peak_coord[1]
        maxx = rangeDoppler[i][coord_R][coord_D]
        thresh = threshold*maxx
        
        # Find the RANGE of values that are within threshold
        idx_1 = 0
        while rangeDoppler[i][coord_R-idx_1][coord_D] > thresh:
            idx_1 += 1
            if coord_R-idx_1 < 0:
                break
        idx_2 = 0
        while rangeDoppler[i][coord_R+idx_2][coord_D] > thresh:
            idx_2 += 1
            if coord_R+idx_2 >= rangeDoppler.shape[1]:
                break
        ranges.append((coord_R-idx_1, coord_R, coord_R+idx_2))
        
    rangeRanges = np.array(ranges)

    uDoppler_stitched = -7*np.ones((rangeDoppler.shape[2],rangeDoppler.shape[0]))

    for i in range(rangeDoppler.shape[0]):
        update_mask = np.sum(rangeDoppler[i,rangeRanges[i,0]:rangeRanges[i,2],:], axis=0)/(rangeRanges[i,2]-rangeRanges[i,0])
        update_mask[update_mask<thresh]=0
        uDoppler_stitched[:,i] += update_mask
    
    return uDoppler_stitched

