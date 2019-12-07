import numpy as np
import func.microdoppler_visualizer as mv
import scipy.io as sio

# =============================================================================
# Load data
# =============================================================================
data = []
trainDataPed = []
trainDataBic = []
trainLabelPed = []
trainLabelBic = []

sets = 5
setSize = 1000

print("[NOTE] ---------- Loading Data ----------")

print("[NOTE] Loading Labels")
testLabelNoCar = sio.loadmat('data/mathworks/test/testLabelNoCar.mat')
testLabelNoCar = testLabelNoCar[list(testLabelNoCar.keys())[-1]].squeeze()
print("[NOTE] Label Loading Complete")

# Grab only Single Pedestrian data
indicesPed = np.where(testLabelNoCar == np.str_('ped    '))[0]     # Get all indices

# Grab only Single Bike data
indicesBic = np.where(testLabelNoCar == np.str_('bic    '))[0]     # Get all indices

print("[NOTE] Loading Time & Frequency Data")
TF = sio.loadmat('data/mathworks/TF.mat')
T = TF[list(TF.keys())[-1]].squeeze()
F = TF[list(TF.keys())[-2]].squeeze()
print("[NOTE] T F Loading Complete")

for i in range(sets):
    testDataNoCar = sio.loadmat('data/mathworks/test/testDataNoCar_'+str(i+1)+'.mat')
    data = np.array(testDataNoCar[list(testDataNoCar.keys())[-1]].squeeze()).transpose(2,0,1)
    print("[NOTE] Loaded data subset ", i)
    trainLabelPed.extend(testLabelNoCar[indicesPed[(indicesPed >= (i*setSize)) & (indicesPed < (i+1)*setSize)]])
    trainDataPed.extend(data[indicesPed[(indicesPed >= (i*setSize)) & (indicesPed < (i+1)*setSize)]-(i*setSize)])
    trainLabelBic.extend(testLabelNoCar[indicesBic[(indicesBic >= (i*setSize)) & (indicesBic < (i+1)*setSize)]])
    trainDataBic.extend(data[indicesBic[(indicesBic >= (i*setSize)) & (indicesBic < (i+1)*setSize)]-(i*setSize)])
    
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
np.save('data/mathworks/test/test_data_ped.npy',trainDataPed)
np.save('data/mathworks/test/test_data_bic.npy',trainDataBic)
np.save('data/mathworks/test/test_label_ped.npy',trainLabelPed)
np.save('data/mathworks/test/test_label_bic.npy',trainLabelBic)
