
import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt

plt.close("all")

def generate_spectrum_from_RDC(filename, numFrames = 500, numADCSamples = 128, numTxAntennas = 3, numRxAntennas = 4, 
                               numLoopsPerFrame = 128, numAngleBins = 64, chirpPeriod = 0.06, logGabor=False, accumulate=True, save_full=False):
    numChirpsPerFrame = numTxAntennas * numLoopsPerFrame
    
# =============================================================================
#     numADCSamples = number of range bins
#     numLoopsPerFrame = number of doppler bins
# =============================================================================

    range_resolution, bandwidth = dsp.range_resolution(numADCSamples)
    doppler_resolution = dsp.doppler_resolution(bandwidth)
    
    if filename[-4:] != '.bin':
        filename += '.bin'
        
    adc_data = np.fromfile(filename, dtype=np.int16)
    adc_data = adc_data.reshape(numFrames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                    num_rx=numRxAntennas, num_samples=numADCSamples)
    print("Data Loaded!")

    dataCube = adc_data
    micro_doppler_data = np.zeros((numFrames, numLoopsPerFrame, numADCSamples), dtype=np.float64)
    theta_data = np.zeros((numFrames, numLoopsPerFrame, numTxAntennas*numRxAntennas,numADCSamples), dtype=np.complex)
    
    for i, frame in enumerate(dataCube):
        # (2) Range Processing
        from mmwave.dsp.utils import Window

        radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
        assert radar_cube.shape == (
        numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

        # (3) Doppler Processing 
        det_matrix , theta_data[i] = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
        
        # --- Shifts & Store
        det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)                
        micro_doppler_data[i,:,:] = det_matrix_vis
        # Data should now be ready. Needs to be in micro_doppler_data, a 3D-numpy array with shape [numDoppler, numRanges, numFrames]
    
        # LOG GABOR
        if logGabor:
            if accumulate:
                image = micro_doppler_data.sum(axis=1).T
            else:
                image = micro_doppler_data.T
    
            from LogGabor import LogGabor
            import holoviews as hv
    
            lg = LogGabor("default_param.py")
            lg.set_size(image)
            lg.pe.datapath = 'database/'
    
            image = lg.normalize(image, center=True)
    
            # display input image
            # hv.Image(image)
    
            # display log gabor'd image
            image = lg.whitening(image)*lg.mask
            hv.Image(image)
    
            uDoppler = image
        elif accumulate:
            uDoppler = micro_doppler_data.sum(axis=1).T
        else:
            uDoppler = micro_doppler_data.T
            
    if save_full:
        return range_resolution, doppler_resolution, uDoppler, theta_data
    else:
        return range_resolution, doppler_resolution, uDoppler