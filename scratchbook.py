# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:11:12 2020

@author: Edwin
"""
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
# Using Radar: Texas Instruments 1843
num_frames = 500
num_adc_samples = 128
num_tx_antennas = 3
num_rx_antennas = 4
num_loops_per_frame = 128
num_chirps_per_frame = num_tx_antennas * num_loops_per_frame

num_range_bins = num_adc_samples
num_doppler_bins = num_loops_per_frame
num_angle_bins = 64
chirp_period = 0.06

range_resolution, bandwidth = dsp.range_resolution(num_adc_samples)
doppler_resolution = dsp.doppler_resolution(bandwidth)

load_data_flag = True

accumulate_flag = True
logGabor_flag = False
plot_range_doppler_flag = False

if __name__ == '__main__':

    if load_data_flag:
        # (1) Reading in adc data
        adc_data = np.fromfile('./data/processed/uDoppler1.bin', dtype=np.int16)
        # adc_data = np.fromfile('./scripts/data/adc_data.bin', dtype=np.int16)
                
        adc_data_padded = np.ones(num_frames*4*num_adc_samples*num_chirps_per_frame*2)*1E-8
        adc_data_padded[:adc_data.shape[0]] = adc_data        
        adc_data = adc_data_padded.reshape(num_frames, -1)
                
        adc_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=num_chirps_per_frame,
                                        num_rx=num_rx_antennas, num_samples=num_adc_samples)
        print("Data Loaded!")
    
        dataCube = adc_data
        
    micro_doppler_data = np.zeros((num_frames, num_loops_per_frame, num_adc_samples), dtype=np.float64)
    for i, frame in enumerate(dataCube):
            # (2) Range Processing
            from mmwave.dsp.utils import Window

            radar_cube = dsp.range_processing(frame, window_type_1d=Window.BLACKMAN)
            assert radar_cube.shape == (
            num_chirps_per_frame, num_rx_antennas, num_adc_samples), "[ERROR] Radar cube is not the correct shape!"

            # (3) Doppler Processing 
            det_matrix , aoa_input = dsp.doppler_processing(radar_cube, num_tx_antennas=3, clutter_removal_enabled=True, window_type_2d=Window.HAMMING)
                        
            # --- Show output
            det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)
            if plot_range_doppler_flag:
                det_matrix_display = det_matrix_vis
                plt.title("Range-Doppler plot " + str(i))
                plt.imshow(det_matrix_display)
                plt.pause(0.05)
                plt.clf()
                
            micro_doppler_data[i,:,:] = det_matrix_vis
            
            
    # Make Cool Plot
    import napari
    viewer = napari.view_image(micro_doppler_data)
    # from mpl_toolkits.mplot3d import Axes3D
    
    # fig = plt.figure()
    
    # ax = fig.gca(projection='3d')
    # x = np.arange(0,micro_doppler_data.shape[0])
    # y = np.arange(0,micro_doppler_data.shape[1])
    # z = np.arange(0,micro_doppler_data.shape[2])
    # ax.scatter(x, y, z)
    
    plt.figure()
    plt.imshow(np.sum(micro_doppler_data, axis=1).T, cmap='turbo',origin='lower',extent=(0,chirp_period*micro_doppler_data[:,120,:].shape[0],-micro_doppler_data[:,120,:].shape[1]*doppler_resolution/2,micro_doppler_data[:,120,:].shape[1]*doppler_resolution/2))

    plt.title("MicroDoppler Ranges Accumulated")
    plt.ylabel("Velocity (m/s)")
    plt.xlabel("Time (seconds)")
    plt.savefig('freq_time_view', bbox_inches = 'tight',pad_inches = 0.25)
    
    # plt.figure()
    # plt.imshow(np.sum(micro_doppler_data, axis=2).T, cmap='turbo',origin='lower',extent=(0,chirp_period*micro_doppler_data[:,120,:].shape[0], 0, micro_doppler_data[:,:,0].shape[1]*range_resolution/2))
