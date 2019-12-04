import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt

plt.close("all")

from func.pca import PCA
from func.ica import ICA
from func.nmf import NMF
from func.generateSpectrogram import generateSpectrogram


# Important Radar Scan Constants
numFrames = 500
numADCSamples = 128
numTxAntennas = 3
numRxAntennas = 4
numLoopsPerFrame = 128
numChirpsPerFrame = numTxAntennas * numLoopsPerFrame

numRangeBins = numADCSamples
numDopplerBins = numLoopsPerFrame
numAngleBins = 64
chirpPeriod = 0.06

range_resolution, bandwidth = dsp.range_resolution(numADCSamples)
doppler_resolution = dsp.doppler_resolution(bandwidth)

LoadData = True
accumulate = True
logGabor = False
plotRangeDoppler = True

if __name__ == '__main__':

    if LoadData:
        # (1) Reading in adc data
#        adc_data = np.fromfile('./data/person_walking_2_Raw_0.bin', dtype=np.uint16)
        adc_data = np.fromfile('./data/uDoppler1.bin', dtype=np.int16)
        adc_data = adc_data.reshape(numFrames, -1)
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=numChirpsPerFrame,
                                        num_rx=numRxAntennas, num_samples=numADCSamples)
        print("Data Loaded!")
    
        dataCube = adc_data

    micro_doppler_data = np.zeros((numFrames, numLoopsPerFrame, numADCSamples), dtype=np.float64)
    for i, frame in enumerate(dataCube):
            # (2) Range Processing
            from mmwave.dsp.utils import Window

            radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
            assert radar_cube.shape == (
            numChirpsPerFrame, numRxAntennas, numADCSamples), "[ERROR] Radar cube is not the correct shape!"

            # (3) Doppler Processing 
            det_matrix , aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
            
#            # Custom signal_2_dB process, bypasses 
#            magnitude_spectrum = np.abs(aoa_input)
#            det_matrix = 20*np.log10(magnitude_spectrum/np.max(magnitude_spectrum))
#            det_matrix = np.sum(det_matrix, axis=1)
#            print("(min,max): (", np.min(det_matrix), ",", np.max(det_matrix),")")
##            det_matrix[det_matrix<-500]=-500
#            threshold = -500
#            det_matrix[det_matrix<threshold]=threshold
#
#
##             Ensure signal is between 0 and 255 (min = -60*12, max = 0)
#            det_matrix = (det_matrix - threshold)*255/(-1*threshold)
            
            # --- Show output
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            if plotRangeDoppler:
#                det_matrix_display = det_matrix_vis / det_matrix_vis.max()
                det_matrix_display = det_matrix_vis
                plt.title("Range-Doppler plot " + str(i))
                plt.imshow(det_matrix_display)
                plt.pause(0.05)
                plt.clf()
                
            micro_doppler_data[i,:,:] = det_matrix_vis
    # Data should now be ready. Needs to be in micro_doppler_data, a 3D-numpy array with shape [numDoppler, numRanges, numFrames]

    # LOG GABOR
    
    if logGabor:
        if accumulate:
            image = micro_doppler_data.sum(axis=1).T
        else:
            image = micro_doppler_data[:,120,:].T

        from LogGabor import LogGabor
        import holoviews as hv
        import os
        fig_width = 12
        figsize=(fig_width, .618*fig_width)

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
        uDoppler = micro_doppler_data[:,80,:].T
    

    plt.figure(1)
    plt.title("micro-Doppler Accumulated")
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (seconds)")
    plt.imshow(uDoppler,origin='lower',extent=(0,chirpPeriod*micro_doppler_data[:,120,:].shape[0],-micro_doppler_data[:,120,:].shape[1]*doppler_resolution/2,micro_doppler_data[:,120,:].shape[1]*doppler_resolution/2))
#    
#    # --- Begin PCA Process ---
#    nFeatures = 3
#    # Calulate mean value over all dimensions
#    mean = np.mean(uDoppler, axis=0)
#    
#    # Center data to 0 mean
#    uDoppler_c = (uDoppler.T-mean[:,None]).T
#    
#    W_features, Z_weights = PCA(uDoppler_c, nFeatures)
#    
#    # Plot Feature vectors to see them
#    time_axis = np.linspace(0,chirpPeriod*uDoppler.shape[1],numFrames)
#    freq_axis = np.linspace(-uDoppler.shape[1]*doppler_resolution/2,uDoppler.shape[1]*doppler_resolution/2, 128)
#        
#    plt.figure(figsize=(14,4))
#    plt.title("Top 3 PCA Feature Components")
#    plt.plot(freq_axis,W_features[0], label='Feature 0')
#    plt.plot(freq_axis,W_features[1], label='Feature 1')
#    plt.plot(freq_axis,W_features[2], label='Feature 2')
#    plt.xlabel("Frequencies (Hz)")
#    plt.grid()
#    plt.legend()
#    
#    fig1 = plt.figure(figsize=(14,10))
#    ax2 = fig1.add_subplot(2,1,1)
#    ax2.set_title("Top 3 PCA Weight Components")
#    ax2.plot(time_axis,Z_weights[0], label='Weight 0')
#    ax2.plot(time_axis,Z_weights[1], label='Weight 1')
#    ax2.plot(time_axis,Z_weights[2], label='Weight 2')
#    ax2.set_xlabel("Time (sec)")
#    ax2.grid()
#    ax2.legend(loc='upper right')
#    
#    ax3 = fig1.add_subplot(2,1,2,sharex=ax2)
#    ax3.imshow(uDoppler,origin='lower',aspect='auto',extent=(0,chirpPeriod*uDoppler.shape[1],-uDoppler.shape[1]*doppler_resolution/2,uDoppler.shape[1]*doppler_resolution/2))
#    
#    W_ICA = ICA(Z_weights, nFeatures, 1E-4, 1E-3)
#    # Calculate new features, and new weights by applying W_ICA as prescribed above
#    W_I = np.linalg.pinv(np.matmul(W_ICA,W_features))  # new Features
#    Z_I = np.matmul(W_I.T,uDoppler_c)  # new Weights
#
#    # Plot ICA
#    plt.figure(figsize=(14,4))
#    plt.title("Top 3 ICA Feature Components")
#    plt.plot(freq_axis,W_I.T[0], label='Feature 0')
#    plt.plot(freq_axis,W_I.T[1], label='Feature 1')
#    plt.plot(freq_axis,W_I.T[2], label='Feature 2')
#    plt.xlabel("Frequencies (Hz)")
#    plt.grid()
#    plt.legend()
#    
#    fig = plt.figure(figsize=(14,10))
#    ax2 = fig.add_subplot(2,1,1)
#    ax2.set_title("Top 3 ICA Weight Components")
#    ax2.plot(time_axis,Z_I[0]+0.002, label='Weight 0')
#    ax2.plot(time_axis,Z_I[1], label='Weight 1')
#    ax2.plot(time_axis,Z_I[2]-0.002, label='Weight 2')
#    ax2.set_xlabel("Time (sec)")
#    ax2.grid()
#    ax2.legend(loc='upper right')
#    
#    ax3 = fig.add_subplot(2,1,2, sharex=ax2)
#    ax3.imshow(uDoppler,origin='lower', aspect='auto', extent=(0,chirpPeriod*uDoppler.shape[1],-uDoppler.shape[1]*doppler_resolution/2,uDoppler.shape[1]*doppler_resolution/2))
#    
#    W_NMF, H_NMF, delta = NMF(uDoppler,3,0.01)
#    
#    plt.figure()
#    plt.plot(delta[1:])
#    plt.show()
#    plt.title('Delta over iterations')
#    
#    # Plot Feature vectors to see them
#    plt.figure(figsize=(14,4))
#    plt.title("Top 3 NMF Feature Components")
#    plt.plot(freq_axis,W_NMF.T[0], label='Feature 0')
#    plt.plot(freq_axis,W_NMF.T[1], label='Feature 1')
#    plt.plot(freq_axis,W_NMF.T[2], label='Feature 2')
#    plt.xlabel("Frequencies (Hz)")
#    plt.grid()
#    plt.legend()
#    
#    fig = plt.figure(figsize=(14,10))
#    ax2 = fig.add_subplot(2,1,1)
#    ax2.set_title("Top 3 NMF Weight Components")
#    ax2.plot(time_axis,H_NMF[0], label='Weight 0')
#    ax2.plot(time_axis,H_NMF[1], label='Weight 1')
#    ax2.plot(time_axis,H_NMF[2], label='Weight 2')
#    ax2.set_xlabel("Time (sec)")
#    ax2.grid()
#    ax2.legend(loc='upper right')
#    
#    ax3 = fig.add_subplot(2,1,2, sharex=ax2)
#    ax3.imshow(uDoppler,origin='lower',aspect='auto', extent=(0,chirpPeriod*uDoppler.shape[1],-uDoppler.shape[1]*doppler_resolution/2,uDoppler.shape[1]*doppler_resolution/2))
