"""
Purpose: This script takes the .mat data files provided by MathWorks and produces .npy data files.

"""

import numpy as np
import func.microdoppler_visualizer as mv
import scipy.io as sio
import matplotlib.pyplot as plt

# =============================================================================
# Load data
# =============================================================================
data = []
trainDataPed = []
trainDataBic = []
trainLabelPed = []
trainLabelBic = []

sets = 19
setSize = 1000

print("[NOTE] ---------- Loading Data ----------")

print("[NOTE] Loading Labels")
trainLabelNoCar = sio.loadmat('data/mathworks/train/trainLabelNoCar.mat')
trainLabelNoCar = trainLabelNoCar[list(trainLabelNoCar.keys())[-1]].squeeze()
print("[NOTE] Label Loading Complete")

# Grab only Single Pedestrian data
indicesPed = np.where(trainLabelNoCar == np.str_('ped    '))[0]     # Get all indices

# Grab only Single Bike data
indicesBic = np.where(trainLabelNoCar == np.str_('bic    '))[0]     # Get all indices

print("[NOTE] Loading Time & Frequency Data")
TF = sio.loadmat('data/mathworks/TF.mat')
T = TF[list(TF.keys())[-1]].squeeze()
F = TF[list(TF.keys())[-2]].squeeze()
print("[NOTE] T F Loading Complete")

for i in range(sets):
    trainDataNoCar = sio.loadmat('data/mathworks/train/trainDataNoCar_'+str(i+1)+'.mat')
    data = np.array(trainDataNoCar[list(trainDataNoCar.keys())[-1]].squeeze()).transpose(2,0,1)
    print("[NOTE] Loaded data subset ", i)
    trainLabelPed.extend(trainLabelNoCar[indicesPed[(indicesPed >= (i*setSize)) & (indicesPed < (i+1)*setSize)]])
    trainDataPed.extend(data[indicesPed[(indicesPed >= (i*setSize)) & (indicesPed < (i+1)*setSize)]-(i*setSize)])
    trainLabelBic.extend(trainLabelNoCar[indicesBic[(indicesBic >= (i*setSize)) & (indicesBic < (i+1)*setSize)]])
    trainDataBic.extend(data[indicesBic[(indicesBic >= (i*setSize)) & (indicesBic < (i+1)*setSize)]-(i*setSize)])
    
    # Check to see the data
    trainDataPed = np.array(trainDataPed)
    mv.classification_data_visualizer(data, trainLabelNoCar)
    assert 1 == 0 , "debug"

trainDataPed = np.array(trainDataPed)
trainLabelPed = np.array(trainLabelPed)
trainDataBic = np.array(trainDataBic)
trainLabelBic = np.array(trainLabelBic)

print("[NOTE] ---------- Data Loading Complete ----------")


## Check to see the data
#mv.classification_data_visualizer(trainDataBic, trainLabelBic)
#
## Check to see the data
#mv.classification_data_visualizer(trainDataPed, trainLabelPed)


# Save as .npy
np.save('data/mathworks/train/train_data_ped.npy',trainDataPed)
np.save('data/mathworks/train/train_data_bic.npy',trainDataBic)
np.save('data/mathworks/train/train_label_ped.npy',trainLabelPed)
np.save('data/mathworks/train/train_label_bic.npy',trainLabelBic)
