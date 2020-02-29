# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:27:16 2020

@author: Edwin
"""

# import
import scipy.io
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

# Load Data
xPedRec = scipy.io.loadmat('xPedRec.mat')['xPedRec']
xBicRec = scipy.io.loadmat('xBicRec.mat')['xBicRec']
Tsamp = 2.815315315315315e-04

# Perform Range FFT
xPedRec_r = fft(xPedRec, axis=0)
xBicRec_r = fft(xBicRec, axis=0)

# Perform STFT
F, T, microdoppler_data_ped = signal.stft(xPedRec_r, 1/Tsamp)
F, T, microdoppler_data_bic = signal.stft(xBicRec_r, 1/Tsamp)

# Grab magnitude
microdoppler_data_ped = np.abs(microdoppler_data_ped)
microdoppler_data_bic = np.abs(microdoppler_data_bic)

microdoppler_data_ped = 10*np.log10(microdoppler_data_ped)
microdoppler_data_bic = 10*np.log10(microdoppler_data_bic)

# Shift Data
F = np.roll(np.flip(F), -len(F)//2)
microdoppler_data_ped = np.roll(np.flip(microdoppler_data_ped, axis=1), -len(F)//2, axis=1)


# Make Cool Plot
import napari
viewer = napari.view_image(microdoppler_data_ped)
# plt.pcolormesh(T, F, np.abs(Zxx))
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()