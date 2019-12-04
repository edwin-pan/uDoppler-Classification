import numpy as np
import mmwave.dsp as dsp
from mmwave.dataloader import DCA1000
import matplotlib.pyplot as plt

plt.close("all")

from func.pca import PCA
from func.generate_spectrum_from_RDC import generate_spectrum_from_RDC
import func.microdoppler_visualizer as mv
import func.range_selection as rs

if __name__ == '__main__':
    plt.close('all')
    
    chirpPeriod = 0.06 # unit?
    numFrames = 500
    range_resolution, doppler_resolution, uDoppler, thetaData = generate_spectrum_from_RDC('./data/uDoppler1.bin', accumulate=False, save_full=True)
    mv.microdoppler_visualizer(uDoppler)
    rangeDoppler = uDoppler.transpose(2,1,0)

    # Select them ranges
    uDoppler_processed = rs.range_doppler_selection(rangeDoppler, threshold=0.75)
#    uDoppler_range_pro = rs.range_selection(rangeDoppler, threshold=0.75)
    
    # Take a look at them ranges
    mv.stitch_visualizer(uDoppler_processed, chirpPeriod, doppler_resolution)
#    mv.stitch_visualizer(uDoppler_range_pro, chirpPeriod, doppler_resolution)
#    mv.stitch_visualizer(np.sum(uDoppler, axis=1), chirpPeriod, doppler_resolution, cmap_plot='viridis')
    
    # Now to get the theta info
    
    
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
#    for i, weight in enumerate(W_features):
#        plt.plot(freq_axis,weight, label='Feature '+str(i))
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
    
